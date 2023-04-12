from tts import *
import knime_extension as knext
import logging

LOGGER = logging.getLogger(__name__)

@knext.node(name="GNN Graph Creator", node_type=knext.NodeType.MANIPULATOR, icon_path="icon.png", category="/")
@knext.input_table(name="Nodes with Features", description="Give a table of ndoes with features and target.")
@knext.input_table(name="Edges", description="Give two columns: one with a node and second column with the paired node")
@knext.output_binary(
    name="Graph",
    description="Pytorch Geometric Graph",
    id="org.knime.torch.graphcreator")
# @knext.output_table("Target Data", "Used for debugging") # DEBUGGING CODE
class GNNCreator:
    """
    Take two tables to construct a full graph. 
    """
    key = knext.ColumnParameter(label="key", description="Index number of each row", port_index=1)
    target = knext.ColumnParameter(label="target", description="The label (y) of each row", port_index=1) #TODO how to select from a list of column
    edge_list_01 = knext.ColumnParameter(label="edge_list_01", description="The first column of edges", port_index=0)
    edge_list_02 = knext.ColumnParameter(label="edge_list_02", description="The second column of edges", port_index=0)
    #TODO add edge features for other kinds of analysis

    def configure(self, configure_context, input_schema_1, input_schema_2):
        return knext.BinaryPortObjectSpec("org.knime.torch.graphcreator")
        # return knext.Schema.from_columns([knext.Column(knext.double(), "Target")])

    def execute(self, exec_context, input_1, input_2):
        edges_data = input_1.to_pandas()
        nodes_data = input_2.to_pandas()
        num_class = nodes_data[self.target].nunique()

        # handle missing values in target variables
        nodes_data[self.target].fillna(num_class-1, inplace=True) #TODO we need to tell the user to transform missing values into 0/0
        
        g = self.construct_graph(nodes_data=nodes_data,
                                 edges_data=edges_data,
                                )

        graph_dict = {'graph':g,
                      'num_of_class':num_class,
                      'key': self.key}

        return pickle.dumps(graph_dict)
        
        # DEBUGGING CODE:
        # labels = list(nodes_data[self.target])
        # output_table = pa.table([pa.array(labels)], names=["Target"])
        # return knext.Table.from_pyarrow(output_table)
    
    def construct_graph(self, nodes_data, edges_data):
        node_features_list = nodes_data.drop(columns=[self.key,self.target]).values.tolist()
        node_features = torch.tensor(node_features_list)
        node_labels = torch.tensor(nodes_data[self.target].values)
        edges_list = edges_data[[self.edge_list_01, self.edge_list_02]].values.tolist()
        edge_index01 = torch.tensor(edges_list, dtype = torch.long).T
        edge_index02 = torch.zeros(edge_index01.shape, dtype = torch.long)
        edge_index02[0,:] = edge_index01[1,:]
        edge_index02[1,:] = edge_index01[0,:]
        edge_index = torch.cat((edge_index01,edge_index02),axis=1)
        g = Data(x=node_features, y=node_labels, edge_index=edge_index)
        return(g)



@knext.node(name="GNN Learner", node_type=knext.NodeType.LEARNER, icon_path="icon.png", category="/")
@knext.input_binary("Full Graph", "Pickled Graph", "org.knime.torch.graphcreator" )
@knext.input_table(name="Train Set", description="A table that contains nodes need to be masked and passed on.")
@knext.output_binary(
    name="Model",
    description="Pytorch Geometric Model",
    id="org.knime.torch.learner",
)
class GNNLearner:
    """
    """
    #TODO Make validation accuracy be in output of the learner node 
    learning_rate = knext.DoubleParameter("Learning Rate", "The learning rate to use in the optimizer", 0.1, min_value=1e-10)
    epochs = knext.IntParameter("Epochs", "The number of epochs to train for.", 1)
    convolutional_layer = knext.StringParameter("Convolutional Layer Choice", "The convolutional layer to be used in the neural network.", "GCNConv", enum=["GCNConv", "SAGEConv"])
    criterion = nn.CrossEntropyLoss()

    def configure(self, configure_context, input_binary_1, input_schema_1):
        return knext.BinaryPortObjectSpec("org.knime.torch.learner")
         
    def execute(self, exec_context, graph_dict, train_set):
        train_set = train_set.to_pandas()
        graph_dict = pickle.loads(graph_dict)

        graph = graph_dict['graph']
        msk = AddMask(graph_dict['key'])
        masked_graph = msk(graph, train_set)

        num_class = graph_dict['num_of_class']
        num_of_feat = masked_graph.num_node_features
        model = GNN(num_of_feat=num_of_feat,
                    hidden_layer=16,
                    num_class=num_class,
                    convolutional_layer=self.convolutional_layer)

        # #### BEGIN TRAINING ####
        # epochs = self.epochs
        # lr = self.learning_rate
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # train_losses=[]
        # train_accuracies=[]
        
        # for e in range(epochs+1):
        #     exec_context.set_progress(progress=e/epochs, message=f"Training {e} for {epochs}")
        #     if exec_context.is_canceled():
        #         raise RuntimeError("Execution terminated by user")
        #     optimizer.zero_grad()
        #     out=model(masked_graph)
        #     loss=self.masked_loss(predictions=out,
        #                           labels=masked_graph.y,
        #                           mask=masked_graph.data_mask)
        #     loss.backward()
        #     optimizer.step()
        #     train_losses+=[loss]
        #     train_accuracy=self.masked_accuracy(predictions=out,
        #                                         labels=masked_graph.y, 
        #                                         mask=masked_graph.data_mask)
        #     train_accuracies+=[float(train_accuracy)] # requires float for pyarrow

        # state_dict = model.state_dict()
        # buffer = BytesIO()
        # torch.save(state_dict, buffer)
        # buffer.seek(0)
        # #### END TRAINING ####

        train_accuracies, buffer = self.train(model, masked_graph, exec_context)

        model_dict = {"model": buffer.read(),
                      "data": graph,
                      "train_accuracies": train_accuracies,
                      "num_of_feat": num_of_feat,
                      "num_of_class": num_class,
                      "key": graph_dict["key"],
                      "convolutional_layer": self.convolutional_layer}
        # statistics_table = pa.table([pa.array([str(i) for i in range(len(train_accuracies))]), pa.array(train_accuracies)], names=["Train Accuracy"])
        
        return pickle.dumps(model_dict)#, knext.Table.from_pyarrow(statistics_table)

    def masked_loss(self, predictions, labels, mask):
        mask=mask.float()
        mask=mask/torch.mean(mask)
        loss=self.criterion(predictions,labels.type(torch.LongTensor))
        loss=loss*mask
        loss=torch.mean(loss)
        return (loss)    

    def masked_accuracy(self, predictions, labels, mask):
        mask=mask.float()
        mask/=torch.mean(mask)
        accuracy=(torch.argmax(predictions,axis=1)==labels).long()
        accuracy=mask*accuracy
        accuracy=torch.mean(accuracy)
        return (accuracy)    

    def train(self, model, g, exec_context):
        epochs = self.epochs
        lr = self.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_losses=[]
        train_accuracies=[]
        
        for e in range(epochs+1):
            exec_context.set_progress(progress=e/epochs, message=f"Training {e} for {epochs}")
            if exec_context.is_canceled():
                raise RuntimeError("Execution terminated by user")
            optimizer.zero_grad()
            out=model(g)
            loss=self.masked_loss(predictions=out,
                                  labels=g.y,
                                  mask=g.data_mask)
            loss.backward()
            optimizer.step()
            train_losses+=[loss]
            train_accuracy=self.masked_accuracy(predictions=out,
                                                labels=g.y, 
                                                mask=g.data_mask)
            train_accuracies+=[float(train_accuracy)] # requires float for pyarrow

        state_dict = model.state_dict()
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        return train_accuracies, buffer



@knext.node(name="GNN Predictor", node_type=knext.NodeType.PREDICTOR, icon_path="icon.png", category="/")
@knext.input_binary("GNN model", "The trained model", "org.knime.torch.learner")
@knext.input_table(name="Test Set", description="A table that contains nodes for testing.")
@knext.output_table("Table with Prediction", "Append prediction probability and class to table")
class GNNPredictor:
    """
    """
    prediction_confidence = knext.BoolParameter("Append overall prediction confidence", "Probability of being the 1 class", False)

    def configure(self, configure_context, input_binary_1, input_schema_1):
        if self.prediction_confidence:
            return knext.Schema.from_columns([knext.Column(knext.double(), "Predictions"), 
                                              knext.Column(knext.double(), "Actual Target"),
                                              knext.Column(knext.double(), "Confidence")])
        else: 
            return knext.Schema.from_columns([knext.Column(knext.double(), "Predictions"), 
                                              knext.Column(knext.double(), "Actual Target")])


    def execute(self, exec_context, model_object, test_set):
        test_set = test_set.to_pandas()
        model_dict = pickle.loads(model_object)
        graph = model_dict["data"]
        num_of_feat = model_dict["num_of_feat"]
        num_class = model_dict["num_of_class"]
        msk = AddMask(model_dict["key"])
        convolutional_layer = model_dict["convolutional_layer"]
        masked_graph = msk(graph, test_set)

        model = GNN(num_of_feat=num_of_feat,
                    hidden_layer=16,
                    num_class=num_class,
                    convolutional_layer=convolutional_layer)

        state_dict = torch.load(BytesIO(model_dict["model"]))
        model.load_state_dict(state_dict)
        model.eval()
        out = model(graph)
       
        key = list(test_set[model_dict["key"]])
        labels = graph.y[masked_graph.data_mask].tolist()
        predictions = (torch.argmax(out,axis=1)[masked_graph.data_mask]).tolist()

        if self.prediction_confidence:
            probabilities = out[masked_graph.data_mask]
            # convert logits output to probability
            probabilities = torch.nn.functional.softmax(probabilities)[ :, 0].tolist()
            output_table = pa.table([pa.array(key), pa.array(labels), pa.array(predictions), pa.array(probabilities)], names=["Key", "Target", "Predictions", "Confidence of being class 0"])
        else:
            output_table = pa.table([pa.array(key), pa.array(labels), pa.array(predictions)], names=["Key", "Target", "Predictions"])

        return knext.Table.from_pyarrow(output_table)
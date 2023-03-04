from tts import *
import knime_extension as knext
import logging

LOGGER = logging.getLogger(__name__)

@knext.node(name="Pytorch Graph Creator", node_type=knext.NodeType.MANIPULATOR, icon_path="icon.png", category="/")
@knext.input_table(name="Nodes with Features", description="Give a table of ndoes with features and target.")
@knext.input_table(name="Edges", description="Give two columns: one with a node and second column with the paired node")
@knext.output_binary(
    name="Graph",
    description="Pytorch Geometric Graph",
    id="org.knime.torch.graphcreator")
class PygCreator:
    """
    Take two tables to construct a full graph. 
    """
    key = knext.ColumnParameter(label="key", description="Index number of each row", port_index=0)
    target_column = knext.ColumnParameter(label="target_column", description="The label (y) of each row", port_index=0)
    edge_list_01 = knext.ColumnParameter(label="edge_list_01", description="The first column of edges", port_index=1)
    edge_list_02 = knext.ColumnParameter(label="edge_list_02", description="The second column of edges", port_index=1)
    #TODO add edge features for other kinds of analysis

    def configure(self, configure_context, input_schema_1, input_schema_2):
        return knext.BinaryPortObjectSpec("org.knime.torch.graphcreator")

    def execute(self, exec_context, input_1, input_2):
        nodes_data = input_1.to_pandas()
        edges_data = input_2.to_pandas()
        
        g = self.construct_graph(nodes_data=nodes_data,
                                edges_data=edges_data,
                                exec=knext.ExecutionContext)

        return pickle.dumps(g)
    
    def construct_graph(self, nodes_data, edges_data, exec:knext.ExecutionContext):
        node_features_list = nodes_data.drop(columns=[self.key,self.target_column]).values.tolist()
        node_features = torch.tensor(node_features_list)
        node_labels = torch.tensor(nodes_data[self.target_column].values)
        edges_list = edges_data[[self.edge_list_01, self.edge_list_02]].values.tolist()
        edge_index01 = torch.tensor(edges_list, dtype = torch.long).T
        edge_index02 = torch.zeros(edge_index01.shape, dtype = torch.long)
        edge_index02[0,:] = edge_index01[1,:]
        edge_index02[1,:] = edge_index01[0,:]
        edge_index = torch.cat((edge_index01,edge_index02),axis=1)
        g = Data(x=node_features, y=node_labels, edge_index=edge_index)
        return(g)


@knext.node(name="Pytorch Mask Creator", node_type=knext.NodeType.MANIPULATOR, icon_path="icon.png", category="/")
@knext.input_binary("Full Graph", "Pickled Graph", "org.knime.torch.graphcreator" )
@knext.input_table(name="Mask Table", description="A table that contains nodes need to be masked and passed on.")
@knext.output_binary(
    name="Masked Graph",
    description="Masked Pytorch Geometric Graph",
    id="org.knime.torch.mask")
class PygMaskCreator:
    """
    Take two tables to construct a full graph. 
    """
    def configure(self, configure_context, input_binary_1, input_schema_2):
        return knext.BinaryPortObjectSpec("org.knime.torch.mask")

    def execute(self, exec_context, graph, table):
        graph = pickle.loads(graph)
        table = table.to_pandas()
        
        msk = AddMask()
        masked_graph = msk(graph, table)

        return pickle.dumps(masked_graph)
    


@knext.node(name="Pytorch GCN learner", node_type=knext.NodeType.LEARNER, icon_path="icon.png", category="/")
@knext.input_binary("Full Masked Graph", "Masked Graph", "org.knime.torch.mask")
@knext.output_binary(
    name="Model",
    description="Pytorch Geometric Model",
    id="org.knime.torch.pygmodel",
)
class PygSplitterLearner:
    """
    Short one-line description of the node.
    Long description of the node.
    Can be multiple lines.
    """
    #TODO Check validation accuracy set be in output of the learner node 
    learning_rate = knext.DoubleParameter("Learning Rate", "The learning rate to use in the optimizer", 0.1, min_value=1e-10)
    epochs = knext.IntParameter("Epochs", "The number of epochs to train for.", 1)
    criterion = nn.CrossEntropyLoss()

    def configure(self, configure_context, input_schema_1):
        return knext.BinaryPortObjectSpec("org.knime.torch.pygmodel")#, knext.Schema.from_columns([knext.Column(knext.string(), "<Row Key>"), knext.Column(knext.double(), "Train Accuracy")])
         
    def execute(self, exec_context, input):
        graph_masked = pickle.loads(input)
        num_of_feat = graph_masked.num_node_features
        model = SocialGNN(num_of_feat=num_of_feat, f=16)

        train_accuracies, buffer = self.train_social(model, graph_masked, knext.ExecutionContext)

        model_dict = {"model": buffer.read(),
                      "data": graph_masked,
                      "num_of_feat": num_of_feat}
        # statistics_table = pa.table([pa.array([str(i) for i in range(len(train_accuracies))]), pa.array(train_accuracies)], names=["Train Accuracy"])
        return pickle.dumps(model_dict)#, knext.Table.from_pyarrow(statistics_table)

    def construct_graph(self, features, edges, target_df, exec: knext.ExecutionContext):
        node_features_list=features.drop(columns=['Key']).values.tolist()
        node_features=torch.tensor(node_features_list)
        node_labels=torch.tensor(target_df['ml_target'].values)
        edges_list=edges.values.tolist()
        edge_index01=torch.tensor(edges_list, dtype = torch.long).T
        edge_index02=torch.zeros(edge_index01.shape, dtype = torch.long)
        edge_index02[0,:]=edge_index01[1,:]
        edge_index02[1,:]=edge_index01[0,:]
        edge_index0=torch.cat((edge_index01,edge_index02),axis=1)
        g = Data(x=node_features, y=node_labels, edge_index=edge_index0)
        return(g)

    def masked_loss(self, predictions, labels, mask, exec: knext.ExecutionContext):
        mask=mask.float()
        mask=mask/torch.mean(mask)
        # had to add the type longtensor
        loss=self.criterion(predictions,labels.type(torch.LongTensor))
        loss=loss*mask
        loss=torch.mean(loss)
        return (loss)    

    def masked_accuracy(self, predictions, labels, mask, exec: knext.ExecutionContext):
        mask=mask.float()
        mask/=torch.mean(mask)
        accuracy=(torch.argmax(predictions,axis=1)==labels).long()
        accuracy=mask*accuracy
        accuracy=torch.mean(accuracy)
        return (accuracy)    

    def train_social(self, model, g, exec: knext.ExecutionContext):
        epochs = self.epochs
        lr = self.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_losses=[]
        train_accuracies=[]
        
        for ep in range(epochs+1):
            optimizer.zero_grad()
            out=model(g)
            loss=self.masked_loss(predictions=out,
                                  labels=g.y,
                                  mask=g.data_mask,
                                  exec=knext.ExecutionContext)
            loss.backward()
            optimizer.step()
            train_losses+=[loss]
            train_accuracy=self.masked_accuracy(predictions=out,
                                                labels=g.y, 
                                                mask=g.data_mask,
                                                exec=knext.ExecutionContext)
            train_accuracies+=[float(train_accuracy)] # requires float for pyarrow

        state_dict = model.state_dict()
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        return train_accuracies, buffer


@knext.node(name="GNN Predictor", node_type=knext.NodeType.PREDICTOR, icon_path="icon.png", category="/")
@knext.input_binary("GNN model", "The trained model", "org.knime.torch.pygmodel")
@knext.input_binary("Graph masked", "Test on market",  "org.knime.torch.mask")
@knext.output_table("Table with Prediction", "Append prediction probability and class to table")
class PygPredictor :
    """
    """
    prediction_confidence = knext.BoolParameter("Append overall prediction confidence", "does not work at the moment", False)

    def configure(self, configure_context, input_schema_1,input_binary_1):
        # return knext.Schema.from_columns([knext.Column(knext.string(), "<Row Key>"), knext.Column(knext.double(), "Test Accuracy")])
        return knext.Schema([knext.double()], ["Table with Prediction"])
 
    def execute(self, exec_context, model_object, mask_data):
        model_dict = pickle.loads(model_object)
        num_of_feat = model_dict["num_of_feat"]
        graph = model_dict["data"]
        model = SocialGNN(num_of_feat=num_of_feat, f=16)
        mask_data = pickle.loads(mask_data)

        state_dict = torch.load(BytesIO(model_dict["model"]))
        model.load_state_dict(state_dict)
        model.eval()
        out = model(graph)

        test_accuracy = self.masked_accuracy(predictions=out,
                                             labels=graph.y, 
                                             mask=mask_data.data_mask,
                                             exec=knext.ExecutionContext)

        test_accuracy = [float(test_accuracy)] # had to add float to work with pyarrow

        statistics_table = pa.table([pa.array(test_accuracy)], names=["Test Accuracy"])
        return knext.Table.from_pyarrow(statistics_table)

    def masked_accuracy(self, predictions, labels, mask, exec: knext.ExecutionContext):
        mask=mask.float()
        mask/=torch.mean(mask)
        accuracy=(torch.argmax(predictions,axis=1)==labels).long()
        accuracy=mask*accuracy
        accuracy=torch.mean(accuracy)
        return (accuracy)    
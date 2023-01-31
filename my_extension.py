from tts import *
import knime_extension as knext
import logging

LOGGER = logging.getLogger(__name__)

@knext.node(name="GNN Partition", node_type=knext.NodeType.MANIPULATOR, icon_path="icon.png", category="/")
@knext.input_table(name="Features of Edges", description="Give a sparse matrix of edges and their features")
@knext.input_table(name="Edges", description="Give two columns: one with a node and second column with the paired node")
@knext.input_table(name="Labels", description="Give two columns: one with input node and second column with binary labels")
# @knext.output_table(
#     name="Train Set",
#     description="Pytorch Geometric Partitioning",
#     id="org.knime.torch.pygpartitioning")
@knext.output_table(
    name="Test Set",
    description="Pytorch Geometric")
class PygPartition:
    """
    Partition GNN into Learner and Predictior
    """
    relative_percentage = knext.DoubleParameter(
                                                label="Percentage",
                                                description="Relative Percentage of Learner Nodes",
                                                default_value=70,
                                                min_value=1,
                                                max_value=100
                                                )

    def configure(self, configure_context, input_schema_1, input_schema_2, input_schema_3):
        return input_schema_1#knext.BinaryPortObjectSpec("org.knime.torch.pygpartitioning")#, knext.Schema([knext.double()], ["Test Accuracy"])
    
    def execute(self, exec_context, input_1, input_2, input_3):
        features  = input_1.to_pandas()
        edges     = input_2.to_pandas()
        target_df = input_3.to_pandas()
        
        # g = self.construct_graph(features=features,
        #                          edges=edges,
        #                          target_df=target_df,
        #                          exec=knext.ExecutionContext)
    
        # msk = AddTrainValTestMask(split="train_rest", num_splits=1, num_val=0.2, num_test=0.3)
        # g = msk(g)
        # train_mask, val_mask, test_mask = msk.__sample_split__(g)
        return input_1#knext.Table.from_pyarrow(train_mask)#, knext.Table.from_pyarrow(test_mask)

@knext.node(name="GNN Learner", node_type=knext.NodeType.LEARNER, icon_path="icon.png", category="/")
@knext.input_table(name="Features of Edges", description="Give a sparse matrix of edges and their features")
@knext.input_table(name="Edges", description="Give two columns: one with a node and second column with the paired node")
@knext.input_table(name="Labels", description="Give two columns: one with input node and second column with binary labels")
@knext.output_binary(
    name="Pyg",
    description="Pytorch Geometric",
    id="org.knime.torch.pyg",
)
# @knext.output_table("Training statistics", "Contains the loss for each epoch.")
class PygSplitterLearner:
    """
    Short one-line description of the node.
    Long description of the node.
    Can be multiple lines.
    """
    learning_rate = knext.DoubleParameter("Learning Rate", "The learning rate to use in the optimizer", 0.1, min_value=1e-10)
    epochs = knext.IntParameter("Epochs", "The number of epochs to train for.", 1)
    criterion = nn.CrossEntropyLoss()

    def configure(self, configure_context, input_schema_1, input_schema_2, input_schema_3):
        return knext.BinaryPortObjectSpec("org.knime.torch.pyg")#, knext.Schema.from_columns([knext.Column(knext.string(), "<Row Key>"), knext.Column(knext.double(), "Train Accuracy")])
         
    def execute(self, exec_context, input_1, input_2, input_3):
        features  = input_1.to_pandas()
        edges     = input_2.to_pandas()
        target_df = input_3.to_pandas()
        
        g = self.construct_graph(features=features,
                                 edges=edges,
                                 target_df=target_df,
                                 exec=knext.ExecutionContext)

        msk = AddTrainValTestMask(split="train_rest", num_splits=1, num_val=0.2, num_test=0.3)
        g = msk(g)

        num_of_feat = g.num_node_features
        model = SocialGNN(num_of_feat=num_of_feat, f=16)

        train_accuracies, buffer = self.train_social(model, g, knext.ExecutionContext)

        model_dict = {
                      "model": buffer.read(),
                      "data": g,
                      "num_of_feat": num_of_feat,
                     }
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

    def train_social(self, model, data, exec: knext.ExecutionContext):
        epochs = self.epochs
        lr = self.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_losses=[]
        train_accuracies=[]
        
        for ep in range(epochs+1):
            optimizer.zero_grad()
            out=model(data)
            loss=self.masked_loss(predictions=out,
                                  labels=data.y,
                                  mask=data.train_mask,
                                  exec=knext.ExecutionContext)
            loss.backward()
            optimizer.step()
            train_losses+=[loss]
            train_accuracy=self.masked_accuracy(predictions=out,
                                                labels=data.y, 
                                                mask=data.train_mask,
                                                exec=knext.ExecutionContext)
            train_accuracies+=[float(train_accuracy)] # requires float for pyarrow

        state_dict = model.state_dict()
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        return train_accuracies, buffer


@knext.node(name="GNN Predictor", node_type=knext.NodeType.PREDICTOR, icon_path="icon.png", category="/")
@knext.input_binary("GNN model", "The trained model", "org.knime.torch.pyg")
@knext.output_table("Test Accuracy", "Contains the accuracy for each epoch.")
class PygPredictor:
    """
    Short one-line description of the node.
    Long description of the node.
    Can be multiple lines.
    """
    prediction_confidence = knext.BoolParameter("Append overall prediction confidence", "does not work at the moment", False)

    def configure(self, configure_context, input_schema_1):
        # return knext.Schema.from_columns([knext.Column(knext.string(), "<Row Key>"), knext.Column(knext.double(), "Test Accuracy")])
        return knext.Schema([knext.double()], ["Test Accuracy"])
 
    def execute(self, exec_context, model_object):
        model_dict = pickle.loads(model_object)
        num_of_feat = model_dict["num_of_feat"]
        data = model_dict["data"]
        model = SocialGNN(num_of_feat=num_of_feat, f=16)

        state_dict = torch.load(BytesIO(model_dict["model"]))
        model.load_state_dict(state_dict)
        model.eval()
        out = model(data)

        test_accuracy = self.masked_accuracy(predictions=out,
                                             labels=data.y, 
                                             mask=data.test_mask,
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
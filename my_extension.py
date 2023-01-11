from tts import *
import knime_extension as knext
import logging

LOGGER = logging.getLogger(__name__)

@knext.node(name="My Template Node", node_type=knext.NodeType.LEARNER, icon_path="icon.png", category="/")
@knext.input_table(name="Features of Edges", description="Give a sparse matrix of edges and their features")
@knext.input_table(name="Edges", description="Give two columns: one with input node and second column with paired node")
@knext.input_table(name="Labels", description="Give two columns: one with input node and second column with binary labels")
@knext.output_table(name="Output", description="Whatever the node has produced")
class PygSplitterLearnerPredictor:
    """
    Short one-line description of the node.
    Long description of the node.
    Can be multiple lines.
    """
    # TODO make some of these work with the code
    learning_rate = knext.DoubleParameter("Learning Rate", "The learning rate to use in the optimizer", 0.1, min_value=1e-10)
    # batch_size = knext.IntParameter("Batch size", "The number of examples to use per batch.", 32, min_value=1)
    epochs = knext.IntParameter("Epochs", "The number of epochs to train for.",2)
    # graph_id = knext.ColumnParameter("Graph identifier column", "The column identifying individual graphs.", column_filter=lambda c: c.ktype == knext.string())
    # node_type_column = knext.ColumnParameter("Node type column", "The column containing the node type in the nodes table.", column_filter=is_int)
    # edge_src_column = knext.ColumnParameter("Edge source column", "Column containing the node index from which an edge originates.", port_index=1, column_filter=is_int)
    # edge_dst_column = knext.ColumnParameter("Edge destination column", "Column containing the node index at which an edge ends.", port_index=1, column_filter=is_int)
    # gru_hs = knext.IntParameter("GRU hidden size", "The hidden size of GRUs", 501, min_value=1)
    # z_dimension = knext.IntParameter("Latent vector size", "The size of the latent vectors.", 56, min_value=1)    
    # node_type_column = knext.ColumnParameter("Node type column", "The column containing the node type in the nodes table.", column_filter=is_int)
    # edge_src_column = knext.ColumnParameter("Edge source column", "Column containing the node index from which an edge originates.", port_index=1, column_filter=is_int)
    # edge_dst_column = knext.ColumnParameter("Edge destination column", "Column containing the node index at which an edge ends.", port_index=1, column_filter=is_int)
    criterion = nn.CrossEntropyLoss()

    def configure(self, configure_context, input_schema_1, input_schema_2, input_schema_3):
    # def configure(self, configure_context, input_schema_1, input_schema_2):  ### Tutorial step 11: Uncomment to configure the new port (and comment out the previous configure header)
        # return knext.Schema.from_columns([knext.Column(knext.double(), "Test Accuracy")])
        return knext.Schema.from_columns([knext.Column(knext.string(), "<Row Key>"), knext.Column(knext.double(), "Test Accuracy")])
        ### Tutorial step 12: Uncomment the following to adjust to the changes we do in this step in the execute method (and comment out the previous return statement)
        # return input_schema_1.append(knext.Column(knext.double(), "column2"))
        ### Tutorial step 13: Uncomment to set a warning for the configuration, which will be shown in the workflow
        # configure_context.set_warning("This is a warning during configuration")
 
    def execute(self, exec_context, input_1, input_2, input_3):
    # def execute(self, exec_context, input_1, input_2):  ### Tutorial step 11: Uncomment to accept the new port (and comment out the previous execute header)
        features = input_1.to_pandas()
        edges = input_2.to_pandas()
        target_df = input_3.to_pandas()
        
        g=self.construct_graph(features=features,
                               edges=edges,
                               target_df=target_df,
                               exec=knext.ExecutionContext)

        msk=AddTrainValTestMask(split="train_rest", num_splits=1, num_val=0.2, num_test=0.3)
        g=msk(g)

        num_of_feat=g.num_node_features
        net=SocialGNN(num_of_feat=num_of_feat, f=16)
        train_accuracies, val_accuracies, test_accuracies = self.train_social(net, g, knext.ExecutionContext)
        # TODO convert this back to pyarrow usage -> pyarrow cannot deal with tensors from pytorch (see error message in next comment)
        # Could not convert tensor(0.2721) with type Tensor: did not recognize Python value type when inferring an Arrow data type 
        statistics_table = pa.table([pa.array([str(i) for i in range(len(test_accuracies))]), pa.array(test_accuracies)], names=["<Row Key>", "Test Accuracy"])
        return knext.Table.from_pyarrow(statistics_table)

        ### Tutorial step 12: Uncomment the following lines to work with the new port (and comment out the previous return statement)
        # input_1_pandas = input_1.to_pandas() # Transform the input table to some processable format (pandas or pyarrow)
        # input_2_pandas = input_2.to_pandas()
        # input_1_pandas['column2'] = input_1_pandas['column1'] + input_2_pandas['column1']
        # return knext.Table.from_pandas(input_1_pandas)
        ### Tutorial step 13: Uncomment the following line to use the parameters from the configuration dialogue (and comment out the previous return statement)
        # input_1_pandas['column2'] = input_1_pandas['column2'] + self.double_param
        # LOGGER.warning(self.double_param) # Tutorial step 14: Logging some warning to the console
        # exec_context.set_warning("This is a warning") # Tutorial step 14: Set a warning to be shown in the workflow
        # return knext.Table.from_pandas(input_1_pandas) ### Tutorial step 13: Uncomment

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

    def train_social(self, net, data, exec: knext.ExecutionContext):
        epochs = self.epochs
        lr = self.learning_rate
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        best_accuracy = 0.0
        
        train_losses=[]
        train_accuracies=[]

        val_losses=[]
        val_accuracies=[]

        test_losses=[]
        test_accuracies=[]
        
        for ep in range(epochs+1):
            optimizer.zero_grad()
            out=net(data)
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
            train_accuracies+=[train_accuracy]
            
            val_loss=self.masked_loss(predictions=out,
                                      labels=data.y, 
                                      mask=data.val_mask,
                                      exec=knext.ExecutionContext)
            val_losses+=[val_loss]
            
            val_accuracy=self.masked_accuracy(predictions=out,
                                              labels=data.y, 
                                              mask=data.val_mask,
                                              exec=knext.ExecutionContext)
            val_accuracies+=[val_accuracy]

            test_accuracy=self.masked_accuracy(predictions=out,
                                               labels=data.y, 
                                               mask=data.test_mask,
                                               exec=knext.ExecutionContext)
            test_accuracies+=[float(test_accuracy)] # had to add float to work with pyarrow
            print("Epoch {}/{}, Train_Loss: {:.4f}, Train_Accuracy: {:.4f}, Val_Accuracy: {:.4f}, Test_Accuracy: {:.4f}"
                    .format(ep+1,epochs, loss.item(), train_accuracy, val_accuracy,  test_accuracy))
            best_accuracy = val_accuracy
        return train_accuracies, val_accuracies, test_accuracies
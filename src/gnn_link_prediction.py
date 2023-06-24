from tts import *
from gnn_link_prediction import *
import knime.extension as knext
import logging

LOGGER = logging.getLogger(__name__)

@knext.node(name="GNN Graph Creator", node_type=knext.NodeType.MANIPULATOR, 
            icon_path="../icons/icon.png", category="/")
@knext.input_table(name="Nodes with Features", 
                   description="Give a table of nodes with features and target.")
@knext.input_table(name="Edges", 
                   description="Give two columns: one with a node and second column with the paired node")
@knext.output_binary(
    name="Graph",
    description="Pytorch Geometric Graph",
    id="org.knime.torch.graphcreator")
class GNNCreator:
    """
    Create a graph to be used fro Graph Nerual Network workflow in kNIME
    Takes two input tables, one for nodes and nodes features, one for edges
    """
    # Define parameters: Nodes Key, Nodes Depend variables, Two columns for Edge connection.
    key = knext.ColumnParameter(label="key", 
                                description="Unique id of each row", port_index=1)
    target = knext.ColumnParameter(label="target", 
                                   description="The label (y) of each row", port_index=1) 
    edge_list_01 = knext.ColumnParameter(label="edge_list_01", 
                                         description="The first column of edges", port_index=0)
    edge_list_02 = knext.ColumnParameter(label="edge_list_02", 
                                         description="The second column of edges", port_index=0)

    def configure(self, configure_context, input_schema_1, input_schema_2):
        return knext.BinaryPortObjectSpec("org.knime.torch.graphcreator")

    def execute(self, exec_context, input_1, input_2):
        """
        Takes two input tables and returns Pytorch Genmetric Graph
        """
        edges_data = input_1.to_pandas()
        nodes_data = input_2.to_pandas()
        num_class = nodes_data[self.target].nunique()

        # Handle missing values in target variables
        nodes_data[self.target].fillna(num_class-1, inplace=True) 
        #TODO we need to tell the user to transform missing values into 0/0
        
        # Construct a grpah object
        g = self.construct_graph(nodes_data=nodes_data,
                                 edges_data=edges_data)

        # Pass graph, saved num_of_class, and Nodes ID column
        graph_dict = {'graph':g,
                      'num_of_class':num_class,
                      'key': self.key}

        return pickle.dumps(graph_dict)
            
    def construct_graph(self, nodes_data, edges_data):
        """
        Constructs a Pytorch Geometric Graph object from the nodes and edges data
        Requires: 
            Nodes with index column and depend variable clolumn
            Edges with two columns representing connections
        """
        # Create Nodes label tensor and Nodes features tensor
        node_features_list = nodes_data.drop(columns=[self.key,self.target]).values.tolist()
        node_features = torch.tensor(node_features_list)
        node_labels = torch.tensor(nodes_data[self.target].values)
        
        # Create edges tensor 
        edges_list = edges_data[[self.edge_list_01, self.edge_list_02]].values.tolist()
        edge_index01 = torch.tensor(edges_list, dtype = torch.long).T
        edge_index02 = torch.zeros(edge_index01.shape, dtype = torch.long)
        edge_index02[0,:] = edge_index01[1,:]
        edge_index02[1,:] = edge_index01[0,:]
        edge_index = torch.cat((edge_index01,edge_index02),axis=1)

        # Construct a graph object from tensors
        g = Data(x=node_features, y=node_labels, edge_index=edge_index)
        return(g)


@knext.node(name="GNN Learner", node_type=knext.NodeType.LEARNER, icon_path="../icons/icon.png", category="/")
@knext.input_binary("Full Graph", "Pickled Graph", "org.knime.torch.graphcreator" )
@knext.input_table(name="Train Set", 
                   description="A table that contains nodes need to be masked and passed downstream.")
@knext.output_binary(
    name="Model",
    description="Pytorch Geometric Model",
    id="org.knime.torch.learner",
)
class GNNLearner:
    """
    Train GNN model based on Graph and pre-selected Nodes
    Takes a graph object and nodes dataframe to generate a trained GNN model
    """
    # Define parameters
    hidden_channels = knext.IntParameter("Hidden Channels", "The number of hidden channels desired.", 1)
    number_of_hidden_layers = knext.IntParameter("Hidden Layers", "The number of hidden layers desired. Recommended to use a number less than the graph diameter if unsure.", 1)
    learning_rate = knext.DoubleParameter("Learning Rate", "The learning rate to use in the optimizer", 0.1, min_value=1e-10)
    epochs = knext.IntParameter("Epochs", "The number of epochs to train for.", 1)
    criterion = nn.CrossEntropyLoss()

    def configure(self, configure_context, input_binary_1, input_schema_1):
        return knext.BinaryPortObjectSpec("org.knime.torch.learner")
         
    def execute(self, exec_context, graph_dict, train_set):
        """
        Train GNN model based on Graph and Nodes dataframe 
        """
        # Convert train_set to a pandas dataframe and graph_dict to a dictionary using pickle
        train_set = train_set.to_pandas()
        graph_dict = pickle.loads(graph_dict)

        # Extract the graph and key from graph_dict, and use AddMask to create a masked version of the graph
        graph = graph_dict['graph']
        msk = AddMask(graph_dict['key'])
        masked_graph = msk(graph, train_set)

        # Extract the number of classes and features from graph_dict, and use them to instantiate a GNN model
        num_class = graph_dict['num_of_class']
        num_of_feat = masked_graph.num_node_features
        model = GNN(in_channels=num_of_feat, 
                     hidden_channels=self.hidden_channels, 
                     out_channels=num_class, 
                     num_layers=self.number_of_hidden_layers)
        
        # Call the train method on the instantiated model and masked graph, and save the returned train_accuracies and buffer
        train_accuracies, buffer = self.train(model, masked_graph, exec_context)
        
        # Create a model_dict dictionary containing relevant information
        model_dict = {"model": buffer.read(),
                      "data": graph,
                      "train_accuracies": train_accuracies,
                      "num_of_feat": num_of_feat,
                      "num_of_class": num_class,
                      "key": graph_dict["key"],
                      "hidden_channels": self.hidden_channels,
                      "number_of_hidden_layers": self.number_of_hidden_layers,
                      }
        # statistics_table = pa.table([pa.array([str(i) for i in range(len(train_accuracies))]), pa.array(train_accuracies)], names=["Train Accuracy"])
        
        return pickle.dumps(model_dict)#, knext.Table.from_pyarrow(statistics_table)

    def masked_loss(self, predictions, labels, mask):
        """
        Calculate loss for the masked nodes
        """
        # Convert the mask to a float tensor and normalize it
        mask=mask.float()
        mask=mask/torch.mean(mask)
        
        # Compute the CrossEntropyLoss between the predictions and labels, masked by the mask tensor
        loss=self.criterion(predictions,labels.type(torch.LongTensor))
        loss=loss*mask
        loss=torch.mean(loss)

        return (loss)    

    def masked_accuracy(self, predictions, labels, mask):
        """
        Calculate accuracy for masked nodes
        """
        # Convert the mask to a float tensor and normalize it
        mask=mask.float()
        mask/=torch.mean(mask)

        # Compute the accuracy as the percentage of correct predictions, masked by the mask tensor
        accuracy=(torch.argmax(predictions,axis=1)==labels).long()
        accuracy=mask*accuracy
        accuracy=torch.mean(accuracy)

        return (accuracy)    

    def train(self, model, g, exec_context):
        """
        Train GNN model with graph
        """
        # Define hyperparameter
        epochs = self.epochs
        lr = self.learning_rate
        
        # Define the optimizer with Adam algorithm
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Create empty lists for training losses and accuracies
        train_losses=[]
        train_accuracies=[]
        
        # Loop over the specified number of epochs
        for e in range(epochs+1):
            # Update the progress bar with the current epoch
            exec_context.set_progress(progress=e/epochs, message=f"Training {e} for {epochs}")
            
            # Check if the user has canceled the execution
            if exec_context.is_canceled():
                raise RuntimeError("Execution terminated by user")
            
            # Remove the perviouse gradients of the optimizer
            optimizer.zero_grad()

            # Get the model's predictions for the input graph
            out = model(data=g)

            # Compute the masked loss using the model's predictions, the graph labels, and the data mask
            loss = self.masked_loss(predictions=out,
                                  labels=g.y,
                                  mask=g.data_mask)
            
            # Compute the gradients of the loss with respect to the model parameters
            loss.backward()
            optimizer.step()
            train_losses+=[loss]
            
            # Compute the masked accuracy using the model's predictions, the graph labels, and the data mask
            train_accuracy=self.masked_accuracy(predictions=out,
                                                labels=g.y, 
                                                mask=g.data_mask)
            
            # Append the current accuracy to the list of training accuracies (requires float for pyarrow)
            train_accuracies+=[float(train_accuracy)] # requires float for pyarrow

        # Save the state dictionary of the model to a buffer
        state_dict = model.state_dict()
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        return train_accuracies, buffer



@knext.node(name="GNN Predictor", node_type=knext.NodeType.PREDICTOR, icon_path="../icons/icon.png", category="/")
@knext.input_binary("GNN model", "The trained model", "org.knime.torch.learner")
@knext.input_table(name="Test Set", description="A table that contains nodes for testing.")
@knext.output_table("Table with Prediction", "Append prediction probability and class to table")
class GNNPredictor:
    """
    A predictor class for Graph Nerual Networks
    Takes trained model and masked graph to generate predictions 
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
        """
        Make predicitons on the input test set based on trained GNN model 
        Takes trained model and test set, returns a table with true values and predictions
        """

        # Convert test_set to a Pandas DataFrame
        test_set = test_set.to_pandas()
        # Load the GNN model dictionary from the saved binary object
        model_dict = pickle.loads(model_object)

        # Extract the GNN model input data and parameters
        data = model_dict["data"]
        num_of_feat = model_dict["num_of_feat"]
        hidden_channels = model_dict["hidden_channels"]
        num_class = model_dict["num_of_class"]
        number_of_hidden_layers = model_dict["number_of_hidden_layers"]
    
        # Create a mask object for masking the input data with the test set
        msk = AddMask(model_dict["key"])
        masked_graph = msk(data, test_set)

        # Create a new GNN model object using the saved parameters
        model = GNN(in_channels=num_of_feat,
                     hidden_channels=hidden_channels,
                     out_channels=num_class, 
                     num_layers=number_of_hidden_layers)

        # Load the saved model state dictionary into the new GNN model
        state_dict = torch.load(BytesIO(model_dict["model"]))
        model.load_state_dict(state_dict)

        # Make predictions using the GNN model on the masked input data
        model.eval()
        out = model(data=data)
       
        # Extract the test set key, actual target, and predicted labels from the GNN model output
        key = list(test_set[model_dict["key"]])
        labels = data.y[masked_graph.data_mask].tolist()
        predictions = (torch.argmax(out,axis=1)[masked_graph.data_mask]).tolist()

        # Add Confidence if needed
        if self.prediction_confidence:
            # Compute the confidence of being class 0 for each prediction
            probabilities = out[masked_graph.data_mask]
            # Convert logits output to probability
            probabilities = torch.nn.functional.softmax(probabilities)[ :, 0].tolist()
            output_table = pa.table([pa.array(key), 
                                     pa.array(labels), 
                                     pa.array(predictions), 
                                     pa.array(probabilities)], 
                                     names=["Key", "Target", "Predictions", "Confidence of being class 0"])
        else:
            output_table = pa.table([pa.array(key), 
                                     pa.array(labels), 
                                     pa.array(predictions)], 
                                     names=["Key", "Target", "Predictions"])

        return knext.Table.from_pyarrow(output_table)
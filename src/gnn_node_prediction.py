from tts import *
import knime_extension as knext
import logging
import utils

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="GNN Heterogeneous Graph Creator",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="icons/icon.png",
    category=utils.category,
)
@knext.input_table(
    name="Heterogeneous Node Connections",
    description="Connect a table with 2 columns - one with first type of node and other with second type of node connected to it.",
)
@knext.input_table(
    name="Node Features",
    description="Connect a table with one node type and all features for that node",
)
@knext.output_binary(
    name="Link Heterogeneous Graph",
    description="Pytorch Geometric Heterogeneous Graph",
    id="org.knime.torch.heterographcreator",
)
class GNNHeteroCreator:
    """
    Takes two tables to construct a full graph.
    """

    node_type_1 = knext.ColumnParameter(
        label="node_type_1", description="Your first node type", port_index=0
    )
    node_type_2 = knext.ColumnParameter(
        label="node_type_2", description="Your second node type", port_index=1
    )

    def configure(self, configure_context, input_schema_1, input_schema_2):
        return knext.BinaryPortObjectSpec("org.knime.torch.heterographcreator")

    def execute(self, exec_context, input_1, input_2):
        node_types = input_1.to_pandas().astype(
            "int64"
        )  # must cast to long type to prevent errors in loss function
        features = input_2.to_pandas().astype(
            "int64"
        )  # must cast to long type to prevent errors in loss function

        # if user only has features for one node type, we need to ensure that the features table is dropping its idx
        # and we need to ensure that the features are being assigned to the correct variable
        if self.node_type_1 not in features.columns:
            self.node_type_1, self.node_type_2 = self.node_type_2, self.node_type_1

        features = torch.from_numpy(
            features.drop([self.node_type_1], axis=1).values
        ).to(torch.float)
        ratings_user_id = torch.from_numpy(
            node_types[self.node_type_2].values
        )  # if successful, change var name
        ratings_movie_id = torch.from_numpy(
            node_types[self.node_type_1].values
        )  # if successful, change var name
        edge_index = torch.stack(
            [ratings_user_id, ratings_movie_id], dim=0
        )  # if successful, change var name in list

        data = HeteroData()

        # Save node indices:
        data["user"].node_id = torch.arange(len(node_types[self.node_type_2].unique()))
        data["movie"].node_id = torch.arange(len(features))

        # Add the node features and edge indices:
        data["movie"].x = features
        data[
            "user", "rates", "movie"
        ].edge_index = edge_index  # if successful, change var names

        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        data = T.ToUndirected()(data)

        graph_dict = {"data": data}

        return pickle.dumps(graph_dict)


@knext.node(
    name="GNN Heterogeneous Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/icon.png",
    category=utils.category,
)
@knext.input_binary("Full Graph", "Pickled Graph", "org.knime.torch.heterographcreator")
@knext.output_binary(
    name="Model",
    description="Pytorch Geometric Model",
    id="org.knime.torch.heterolinklearner",
)
@knext.output_table("Training Loss", "Training Loss")
class GNNHeteroLinkLearner:
    hidden_channels = knext.IntParameter(
        "Hidden Channels", "The number of hidden channels desired.", 1
    )
    number_of_hidden_layers = knext.IntParameter(
        "Hidden Layers",
        "The number of hidden layers desired. Recommended to use a number less than the graph diameter if unsure.",
        1,
    )
    learning_rate = knext.DoubleParameter(
        "Learning Rate",
        "The learning rate to use in the optimizer",
        0.001,
        min_value=1e-10,
    )
    epochs = knext.IntParameter("Epochs", "The number of epochs to train for.", 1)

    def configure(self, configure_context, input_binary_1):
        return knext.BinaryPortObjectSpec(
            "org.knime.torch.heterolinklearner"
        ), knext.Schema.from_columns(
            [
                knext.Column(knext.double(), "Epoch"),
                knext.Column(knext.double(), "Loss"),
            ]
        )

    def execute(self, exec_context, graph_dict):
        graph_dict = pickle.loads(graph_dict)

        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=("user", "rates", "movie"),
            rev_edge_types=("movie", "rev_rates", "user"),
        )

        data = graph_dict["data"]
        train_data, val_data, test_data = transform(data)

        # Define seed edges:
        edge_label_index = train_data["user", "rates", "movie"].edge_label_index
        edge_label = train_data["user", "rates", "movie"].edge_label

        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=(("user", "rates", "movie"), edge_label_index),
            edge_label=edge_label,
            batch_size=128,
            shuffle=True,
        )

        model = Model(data=data, hidden_channels=64)
        buffer, output_table = self.train(model, train_loader, exec_context)

        model_dict = {
            "model": buffer.read(),
            "data": data,
            "hidden_channels": self.hidden_channels,
            "val_data": val_data,
        }

        return pickle.dumps(model_dict), knext.Table.from_pyarrow(output_table)

    def train(self, model, train_loader, exec_context):
        epochs = self.epochs
        lr = self.learning_rate
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        loss_list = []
        for e in range(epochs + 1):
            exec_context.set_progress(
                progress=e / epochs, message=f"Training {e} for {epochs}"
            )
            if exec_context.is_canceled():
                raise RuntimeError("Execution terminated by user")
            total_loss = total_examples = 0
            for sampled_data in train_loader:
                sampled_data.to(device)
                optimizer.zero_grad()
                pred = model(sampled_data)
                ground_truth = sampled_data["user", "rates", "movie"].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            loss_list.append(round(total_loss / total_examples, 4))

        state_dict = model.state_dict()
        buffer = BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        output_table = pa.table(
            [pa.array([str(i) for i in range(len(loss_list))]), pa.array(loss_list)],
            names=["Epoch", "Loss"],
        )

        return buffer, output_table


@knext.node(
    name="GNN Heterogeneous Predictor",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/icon.png",
    category=utils.category,
)
@knext.input_binary(
    "GNN model", "The trained model", "org.knime.torch.heterolinklearner"
)
@knext.output_table(
    "Table with Prediction", "Append prediction probability and class to table"
)
class GNNHeteroLinkPredictor:
    """ """

    prediction_confidence = knext.BoolParameter(
        "Append overall prediction confidence",
        "Probability of being the 1 class",
        False,
    )

    def configure(self, configure_context, input_binary_1):
        if self.prediction_confidence:
            return knext.Schema.from_columns(
                [
                    knext.Column(knext.double(), "Predictions"),
                    knext.Column(knext.double(), "Actual Target"),
                    knext.Column(knext.double(), "Confidence"),
                ]
            )
        else:
            return knext.Schema.from_columns(
                [
                    knext.Column(knext.double(), "Predictions"),
                    knext.Column(knext.double(), "Actual Target"),
                ]
            )

    def execute(self, exec_context, model_object):
        model_dict = pickle.loads(model_object)
        data = model_dict["data"]
        # hidden_channels = model_dict["hidden_channels"]
        # number_of_hidden_layers = model_dict["number_of_hidden_layers"]

        model = Model(data=data, hidden_channels=64)

        state_dict = torch.load(BytesIO(model_dict["model"]))
        model.load_state_dict(state_dict)
        val_data = model_dict["val_data"]
        # model.eval()

        # Define the validation seed edges:
        edge_label_index = val_data["user", "rates", "movie"].edge_label_index
        edge_label = val_data["user", "rates", "movie"].edge_label

        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=[20, 10],
            edge_label_index=(("user", "rates", "movie"), edge_label_index),
            edge_label=edge_label,
            batch_size=3 * 128,
            shuffle=False,
        )

        preds = []
        ground_truths = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for sampled_data in val_loader:
            with torch.no_grad():
                sampled_data.to(device)
                preds.append(model(sampled_data))
                ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        # if self.prediction_confidence:
        #     probabilities = out[masked_graph.data_mask]
        #     probabilities = torch.nn.functional.softmax(probabilities)[ :, 0].tolist()
        #     output_table = pa.table([pa.array(key),
        #                              pa.array(labels),
        #                              pa.array(predictions),
        #                              pa.array(probabilities)],
        #                              names=["Key", "Target", "Predictions", "Confidence of being class 0"])
        # else:
        output_table = pa.table(
            [
                pa.array([str(i) for i in range(len(ground_truth))]),
                pa.array(ground_truth),
                pa.array(pred),
            ],
            names=["Id", "Ground Truth", "Prediction"],
        )

        return knext.Table.from_pyarrow(output_table)

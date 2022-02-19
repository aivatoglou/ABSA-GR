import torch
import torch.nn as nn
from capsule_layer import CapsuleLinear


class BERTGRUSentiment(nn.Module):

    """The hybrid architecture.

    Consists of the Greek-BERT, three bi-GRUs, a CapsNet and four fully connected layers in total.
    """

    def __init__(self, bert, n_classes, batch_size):

        super().__init__()

        self.bert = bert
        self.embedding_dim = bert.config.to_dict()["hidden_size"]  # 768
        self.n_classes = n_classes

        # GRU
        self.hidden = 512
        self.n_layers = 2
        self.batch_size = batch_size

        # CapsNet
        self.in_length = 128
        self.out_length = 8
        self.share_weight = True
        self.routing_type = "dynamic"
        self.num_iterations = 5

        self.gru1 = nn.GRU(
            self.embedding_dim * 4,
            self.hidden,
            self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if self.n_layers == 1 else 0.25,
        )

        self.gru2 = nn.GRU(
            self.embedding_dim * 4,
            self.hidden,
            self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if self.n_layers == 1 else 0.25,
        )

        self.gru3 = nn.GRU(
            self.embedding_dim * 4,
            self.hidden,
            self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if self.n_layers == 1 else 0.25,
        )

        self.dropout = nn.Dropout(0.5)
        self.caps_net = CapsuleLinear(
            out_capsules=self.n_classes,
            in_length=self.in_length,
            out_length=self.out_length,
            share_weight=self.share_weight,
            routing_type=self.routing_type,
            num_iterations=self.num_iterations,
        )
        self.fc = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, self.n_classes)

        self.caps_fc = nn.Linear(self.hidden, self.in_length)

    def forward(self, text, ids, mask):

        # sequence_output.shape = [128, 210, 768], pooled_output.shape = [128, 768], hidden_states = tuple()
        sequence_output, pooled_output, hidden_states = self.bert(
            input_ids=text, token_type_ids=ids, attention_mask=mask, return_dict=False
        )

        embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]

        # Concat batches of 4 hidden layers of BERT
        pooled_output_1 = torch.cat(
            tuple([attention_hidden_states[i] for i in [-12, -11, -10, -9]]), dim=-1
        )  # (64, 210, 3072)
        pooled_output_2 = torch.cat(
            tuple([attention_hidden_states[i] for i in [-8, -7, -6, -5]]), dim=-1
        )  # (64, 210, 3072)
        pooled_output_3 = torch.cat(
            tuple([attention_hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1
        )  # (64, 210, 3072)

        out1, hidden1 = self.gru1(pooled_output_1)  # (4, 64, 512)
        out2, hidden2 = self.gru2(pooled_output_2)  # (4, 64, 512)
        out3, hidden3 = self.gru3(pooled_output_3)  # (4, 64, 512)

        hidden1 = hidden1.view(self.n_layers, 2, self.batch_size, self.hidden)
        hidden1 = hidden1[-1]  # (2, 64, 512)

        hidden2 = hidden2.view(self.n_layers, 2, self.batch_size, self.hidden)
        hidden2 = hidden2[-1]  # (2, 64, 512)

        hidden3 = hidden3.view(self.n_layers, 2, self.batch_size, self.hidden)
        hidden3 = hidden3[-1]  # (2, 64, 512)

        hidden_concat = torch.cat((hidden1, hidden2, hidden3), dim=0)  # (6, 64, 512)

        ############################### CapsNet ###############################

        # Average of hidden layers
        caps_input = torch.mean(hidden_concat, 0)  # (64, 512)

        # Dropout layer
        caps_input = self.dropout(caps_input)

        # Fully connected
        caps_input = self.caps_fc(caps_input)  # (64, 128)

        # We need an extra dimension for CapsNet
        caps_input = caps_input.unsqueeze(1)  # (64, 1 , 128)

        # Dropout layer
        caps_input = self.dropout(caps_input)

        # Capsule classifier
        caps_input = caps_input.to(torch.float32).contiguous()
        caps_output, caps_prob = self.caps_net(caps_input)  # (64, 3, 8)

        # Classification
        caps_output = caps_output.norm(dim=-1)  # (64, 3)

        ########################### Fully connected ###########################

        # Average of hidden layers
        fc_input = torch.mean(hidden_concat, 0)  # (64, 512)

        # Dropout layer
        fc_input = self.dropout(fc_input)

        # Fully connected layer
        fc_input = self.fc(fc_input)  # (64, 128)

        # Dropout layer
        fc_input = self.dropout(fc_input)

        # Fully connected layer
        fc_input = self.fc2(fc_input)  # (64, 32)

        # Dropout layer
        fc_input = self.dropout(fc_input)

        # Fully connected layer
        fc_output = self.fc3(fc_input)  # (64, 3)

        ############################# Soft Voting #############################

        # Soft voting
        ensemble_predictions = (caps_output + fc_output) / 2.0

        return ensemble_predictions

import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GravNetConv

from .utils import CLUSTERS_X, TRACKS_X


class VICReg(nn.Module):
    def __init__(self, encoder, decoder):
        super(VICReg, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def distinguish_PFelements(self, batch):
        """Takes an event~Batch() and splits it into two Batch() objects representing the tracks/clusters."""

        track_id = 1
        cluster_id = 2

        tracks = Batch(
            x=batch.x[batch.x[:, 0] == track_id][:, 1:].float()[
                :, :TRACKS_X
            ],  # remove the first input feature which is not needed anymore
            ygen=batch.ygen[batch.x[:, 0] == track_id],
            ygen_id=batch.ygen_id[batch.x[:, 0] == track_id],
            ycand=batch.ycand[batch.x[:, 0] == track_id],
            ycand_id=batch.ycand_id[batch.x[:, 0] == track_id],
            batch=batch.batch[batch.x[:, 0] == track_id],
        )
        clusters = Batch(
            x=batch.x[batch.x[:, 0] == cluster_id][:, 1:].float()[
                :, :CLUSTERS_X
            ],  # remove the first input feature which is not needed anymore
            ygen=batch.ygen[batch.x[:, 0] == cluster_id],
            ygen_id=batch.ygen_id[batch.x[:, 0] == cluster_id],
            ycand=batch.ycand[batch.x[:, 0] == cluster_id],
            ycand_id=batch.ycand_id[batch.x[:, 0] == cluster_id],
            batch=batch.batch[batch.x[:, 0] == cluster_id],
        )
        return tracks, clusters

    def forward(self, event):

        # seperate tracks from clusters
        tracks, clusters = self.distinguish_PFelements(event)

        # encode to retrieve the representations
        track_representations, cluster_representations = self.encoder(tracks, clusters)

        # decode/expand to get the embeddings
        embedding_tracks, embedding_clusters = self.decoder(track_representations, cluster_representations)

        # global pooling to be able to compute a loss
        pooled_tracks = global_mean_pool(embedding_tracks, tracks.batch)
        pooled_clusters = global_mean_pool(embedding_clusters, clusters.batch)

        return pooled_tracks, pooled_clusters


class ENCODER(nn.Module):
    """The Encoder part of VICReg which attempts to learns useful latent representations of tracks and clusters."""

    def __init__(
        self,
        width=126,
        embedding_dim=34,
        num_convs=2,
        space_dim=4,
        propagate_dim=22,
        k=8,
    ):
        super(ENCODER, self).__init__()

        self.act = nn.ELU

        # 1. different embedding of tracks/clusters
        self.nn1 = nn.Sequential(
            nn.Linear(TRACKS_X, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, embedding_dim),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(CLUSTERS_X, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, embedding_dim),
        )

        # 2. same GNN for tracks/clusters
        self.conv = nn.ModuleList()
        for i in range(num_convs):
            self.conv.append(
                GravNetConv(
                    embedding_dim,
                    embedding_dim,
                    space_dimensions=space_dim,
                    propagate_dimensions=propagate_dim,
                    k=k,
                )
            )

    def forward(self, tracks, clusters):

        embedding_tracks = self.nn1(tracks.x.float())
        embedding_clusters = self.nn2(clusters.x.float())

        # perform a series of graph convolutions
        for num, conv in enumerate(self.conv):
            embedding_tracks = conv(embedding_tracks, tracks.batch)
            embedding_clusters = conv(embedding_clusters, clusters.batch)

        return embedding_tracks, embedding_clusters


class DECODER(nn.Module):
    """The Decoder part of VICReg which attempts to expand the learned latent representations
    of tracks and clusters into a space where a loss can be computed."""

    def __init__(
        self,
        input_dim=34,
        width=126,
        output_dim=200,
    ):
        super(DECODER, self).__init__()

        self.act = nn.ELU

        # DECODER
        self.expander = nn.Sequential(
            nn.Linear(input_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, output_dim),
        )

    def forward(self, out_tracks, out_clusters):

        return self.expander(out_tracks), self.expander(out_clusters)

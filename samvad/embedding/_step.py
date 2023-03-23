from dataclasses import dataclass

import numpy as np
from torch import Tensor

from samvad.core import api, presets

from ._base import generate_embedding_amlv, generate_embedding_ambv, generate_embedding_openai, generate_embedding_gpt2
from ._to_index import embedding_to_indexes_faiss, embedding_to_indexes_annoy, embedding_to_indexes_milvus,\
                       embedding_to_indexes_hnswlib


@dataclass
class EmbeddingOutput(api.Output):
    tensor: Tensor


@dataclass
class EmbeddingToIndexesContext(api.Context):
    embeddings: np.ndarray


@dataclass
class EmbeddingToIndexesOutput(api.Output):
    embeddings: np.ndarray


class EmbeddingAmlvStep(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context()
        output: Tensor = generate_embedding_amlv(context.input_text)
        self.set_output(output=EmbeddingOutput(tensor=output))


class EmbeddingAmbvStep(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context()
        output: Tensor = generate_embedding_ambv(context.input_text)
        self.set_output(output=EmbeddingOutput(tensor=output))


class EmbeddingOpenaiStep(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context()
        output: Tensor = generate_embedding_openai(context.input_text)
        self.set_output(output=EmbeddingOutput(tensor=output))


class EmbeddingGpt2Step(api.Step):
    def run(self) -> None:
        context: presets.StringGeneralContext = self.get_context()
        output: Tensor = generate_embedding_gpt2(context.input_text)
        self.set_output(output=EmbeddingOutput(tensor=output))


class EmbeddingToIndexesFaissStep(api.Step):
    def run(self) -> None:
        context: EmbeddingToIndexesContext = self.get_context(EmbeddingToIndexesContext)
        output: np.ndarray = embedding_to_indexes_faiss(context.embeddings)
        self.set_output(output=EmbeddingToIndexesOutput(embeddings=output))


class EmbeddingToIndexesAnnoyStep(api.Step):
    def run(self) -> None:
        context: EmbeddingToIndexesContext = self.get_context(EmbeddingToIndexesContext)
        output: np.ndarray = embedding_to_indexes_annoy(context.embeddings)
        self.set_output(output=EmbeddingToIndexesOutput(embeddings=output))


class EmbeddingToIndexesMilvusStep(api.Step):
    def run(self) -> None:
        context: EmbeddingToIndexesContext = self.get_context(EmbeddingToIndexesContext)
        output: np.ndarray = embedding_to_indexes_milvus(context.embeddings, '')
        self.set_output(output=EmbeddingToIndexesOutput(embeddings=output))


class EmbeddingToIndexesHnswlibStep(api.Step):
    def run(self) -> None:
        context: EmbeddingToIndexesContext = self.get_context(EmbeddingToIndexesContext)
        output: np.ndarray = embedding_to_indexes_hnswlib(context.embeddings)
        self.set_output(output=EmbeddingToIndexesOutput(embeddings=output))

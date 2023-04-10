from samvad.core import casting, presets
from ._step import EmbeddingToIndexesOutput, EmbeddingOutput, EmbeddingToIndexesContext


class CastEmbeddingTensorToNdArray(casting.CastOutputToContext):
    def cast(self, context: presets.StringGeneralContext, output: EmbeddingOutput) -> EmbeddingToIndexesContext:
        """
            Casts a PyTorch tensor to a numpy ndarray.

            Args:
            - x (torch.Tensor): The input tensor to cast.
            - use_gpu (bool): A flag indicating whether to use GPU or not.

            Returns:
            - A numpy ndarray with the same shape and data as the input tensor.
            """
        use_gpu: bool = False
        if use_gpu:
            x = output.tensor.detach().cpu()
        return x.detach().numpy()


class CastEmbeddingNdArrayToText(casting.CastOutputToContext):
    def cast(self, context: EmbeddingToIndexesContext, output: EmbeddingToIndexesOutput) -> presets.StringGeneralContext:
        pass

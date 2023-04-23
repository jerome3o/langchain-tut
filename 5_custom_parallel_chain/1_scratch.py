from typing import List, Dict

from langchain.chains import LLMChain
from langchain.chains.base import Chain


class SplitLLMChain(Chain):
    def __init__(self, inner_chain: LLMChain):
        super().__init__()

        self._inner_chain = inner_chain

    @property
    def input_keys(self) -> List[str]:
        return self._inner_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        return self._inner_chain.output_keys

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        pass

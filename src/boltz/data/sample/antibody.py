from dataclasses import replace
from typing import Iterator, List

from numpy.random import RandomState

from boltz.data.types import Record, AntibodyInfo
from boltz.data.sample.sampler import Sample, Sampler


class AntibodySampler(Sampler):
    """A simple random sampler with replacement."""

    def sample(self, records: List[Record], random: RandomState) -> Iterator[Sample]:
        """Sample a structure from the antibody dataset infinitely.

        Parameters
        ----------
        records : List[Record]
            The records to sample from.
        random : RandomState
            The random state for reproducibility.

        Yields
        ------
        Sample
            A data sample.

        """
        items = []
        for record in records:
            assert isinstance(record.structure, AntibodyInfo)
            h_chain_id = record.structure.H_chain_id
            l_chain_id = record.structure.L_chain_id
            if h_chain_id is not None and record.chains[h_chain_id].valid:
                items.append((record, h_chain_id))
            if l_chain_id is not None and record.chains[l_chain_id].valid:
                items.append((record, l_chain_id))
        
        while True:
            item_idx = random.randint(len(items))
            record, index = items[item_idx]
            yield Sample(record=record, chain_id=index)
            

    
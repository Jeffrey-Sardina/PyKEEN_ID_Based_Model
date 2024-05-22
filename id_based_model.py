import torch
from torch import nn
import pykeen
from pykeen.nn.modules import Interaction, parallel_unsqueeze
import torch.nn.functional as F
from pykeen.nn.representation import Representation
from typing import Optional, Sequence
from pykeen.nn import Embedding

class ID_Based_Interaction(Interaction):
    '''
    Note: this is very similar to ERMLP
    (see: https://pykeen.readthedocs.io/en/stable/byo/interaction.html)
    '''
    def __init__(
            self,
            *args,
            **kwargs
        ):
        super().__init__()
        # your code here...
    
    def update(
            self,
            _get_representations=None
        ):
        assert _get_representations, f'_get_representations must have a valud but is {_get_representations}'
        self._get_representations = _get_representations

    def forward(
            self,
            mode=None,
            h_ids=None,
            r_ids=None,
            t_ids=None,
            **kwargs
        ):
        # get embeddings from the IDs
        h, r, t = self._get_representations(
            h=h_ids,
            r=r_ids,
            t=t_ids,
            mode=mode
        )

        # compute a score using the embeddings
        return -(h + r - t).norm(p=2, dim=-1)
    
    def score_hrt(
            self,
            mode=None,
            h_ids=None,
            r_ids=None,
            t_ids=None,
            **kwargs
        ):
        scores = self.forward(
            mode=mode,
            h_ids=h_ids,
            r_ids=r_ids,
            t_ids=t_ids,
            **kwargs
        )
        scores = torch.unsqueeze(scores, 1)
        return scores

def repeat_if_necessary(
    scores: torch.FloatTensor,
    representations: Sequence[Representation],
    num: Optional[int],
) -> torch.FloatTensor:
    '''
    copised from https://pykeen.readthedocs.io/en/stable/_modules/pykeen/models/nbase.html#ERModel
    '''
    if representations:
        return scores
    return scores.repeat(1, num)

class ID_Based_Model(pykeen.models.multimodal.base.ERModel):
    def __init__(
        self,
        dim: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(
            interaction=ID_Based_Interaction,
            interaction_kwargs={
                # ... your args here ... 
            },
            entity_representations=Embedding,
            entity_representations_kwargs=dict(
                embedding_dim=dim,
            ),
            relation_representations=Embedding,
            relation_representations_kwargs=dict(
                embedding_dim=dim,
            ),
            **kwargs,
        )

        # we can't init the interaction with these since bc that would be a
        # circuclar dependency, so instead we immediately pass them to an
        # update function
        self.interaction.update(
            _get_representations=self._get_representations,
        )

    def forward(
        self,
        h_indices: torch.LongTensor,
        r_indices: torch.LongTensor,
        t_indices: torch.LongTensor,
        slice_size: Optional[int] = None,
        slice_dim: int = 0,
        *,
        mode = None
    ) -> torch.FloatTensor:      
        if not self.entity_representations or not self.relation_representations:
            raise NotImplementedError("repeat scores not implemented for general case.")
        return self.interaction.score(
            mode=mode,
            h_ids=h_indices,
            r_ids=r_indices,
            t_ids=t_indices,
            slice_size=slice_size,
            slice_dim=slice_dim,
        )
    
    def score_hrt(
            self,
            hrt_batch: torch.LongTensor,
            *,
            mode = None
        ) -> torch.FloatTensor:
        return self.interaction.score_hrt(
            mode=mode,
            h_ids=hrt_batch[:, 0],
            r_ids=hrt_batch[:, 1],
            t_ids=hrt_batch[:, 2],
        )
    
    def score_h(
        self,
        rt_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode = None,
        heads: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # normalize before checking
        if slice_size and slice_size >= self.num_entities:
            slice_size = None
        self._check_slicing(slice_size=slice_size)

        # slice early to allow lazy computation of target representations
        if slice_size:
            return torch.cat(
                [
                    self.score_h(
                        rt_batch=rt_batch,
                        slice_size=None,
                        mode=mode,
                        heads=torch.arange(start=start, end=min(start + slice_size, self.num_entities)),
                    )
                    for start in range(0, self.num_entities, slice_size)
                ],
                dim=-1,
            )

        # add broadcast dimension
        rt_batch = rt_batch.unsqueeze(dim=1)

        # keep these as IDs, not embedding vectors
        if not heads:
            heads = torch.arange(start=0, end=self.num_entities)

        # unsqueeze if necessary
        if heads is None or heads.ndimension() == 1:
            heads = parallel_unsqueeze(heads, dim=1)

        h = heads
        r = rt_batch[..., 0]
        t = rt_batch[..., 1]

        assert r.shape == t.shape
        num_rt_pairs = r.shape[0]
        num_heads = h.shape[0]

        '''
        Repeat interleave on h so all h's get mapped to all rt pairs
        https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html
        '''
        h = h.repeat_interleave(num_rt_pairs, dim=0)
        r = r.repeat(num_heads, 1)
        t = t.repeat(num_heads, 1)

        scores = repeat_if_necessary(
            scores=self.interaction(
                mode=mode,
                h_ids=h,
                r_ids=r,
                t_ids=t
            ),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode) if heads is None else heads.shape[-1],
        )
        scores = scores.reshape(-1, num_heads)
        return scores

    def score_r(
        self,
        ht_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode = None,
        relations: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # normalize before checking
        if slice_size and slice_size >= self.num_relations:
            slice_size = None
        self._check_slicing(slice_size=slice_size)

        # slice early to allow lazy computation of target representations
        if slice_size:
            return torch.cat(
                [
                    self.score_r(
                        ht_batch=ht_batch,
                        slice_size=None,
                        mode=mode,
                        relations=torch.arange(start=start, end=min(start + slice_size, self.num_relations)),
                    )
                    for start in range(0, self.num_relations, slice_size)
                ],
                dim=-1,
            )

        # add broadcast dimension
        ht_batch = ht_batch.unsqueeze(dim=1)
        
        # keep these as IDs, not embedding vectors
        if not relations:
            relations = torch.arange(start=0, end=self.num_relations)

        # unsqueeze if necessary
        if relations is None or relations.ndimension() == 1:
            relations = parallel_unsqueeze(relations, dim=1)

        h = ht_batch[..., 0]
        r = relations
        t = ht_batch[..., 1]

        assert h.shape == t.shape
        num_ht_pairs = h.shape[0]
        num_relations = r.shape[0]
        
        '''
        Repeat interleave on r so all r's get mapped to all ht pairs
        https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html
        '''
        h = h.repeat(num_relations, 1)
        r = r.repeat_interleave(num_ht_pairs, dim=0)
        t = t.repeat(num_relations, 1)

        scores = repeat_if_necessary(
            scores=self.interaction(
                mode=mode,
                h_ids=h,
                r_ids=r,
                t_ids=t
            ),
            representations=self.relation_representations,
            num=self.num_relations if relations is None else relations.shape[-1],
        )
        scores = scores.reshape(-1, num_relations)
        return scores

    def score_t(
        self,
        hr_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode = None,
        tails: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # normalize before checking
        if slice_size and slice_size >= self.num_entities:
            slice_size = None
        self._check_slicing(slice_size=slice_size)

        # slice early to allow lazy computation of target representations
        if slice_size:
            return torch.cat(
                [
                    self.score_t(
                        hr_batch=hr_batch,
                        slice_size=None,
                        mode=mode,
                        tails=torch.arange(start=start, end=min(start + slice_size, self.num_entities)),
                    )
                    for start in range(0, self.num_entities, slice_size)
                ],
                dim=-1,
            )

        # add broadcast dimension
        hr_batch = hr_batch.unsqueeze(dim=1)

        # keep these as IDs, not embedding vectors
        if not tails:
            tails = torch.arange(start=0, end=self.num_entities)
        
        # unsqueeze if necessary
        if tails is None or tails.ndimension() == 1:
            tails = parallel_unsqueeze(tails, dim=1)

        h = hr_batch[..., 0]
        r = hr_batch[..., 1]
        t = tails

        assert h.shape == r.shape
        num_hr_pairs = r.shape[0]
        num_tails = t.shape[0]

        '''
        Repeat interleave on t so all t's get mapped to all hr pairs
        https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html
        '''
        h = h.repeat(num_tails, 1)
        r = r.repeat(num_tails, 1)
        t = t.repeat_interleave(num_hr_pairs, dim=0)

        scores = repeat_if_necessary(
            scores=self.interaction(
                mode=mode,
                h_ids=h,
                r_ids=r,
                t_ids=t
            ),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode) if tails is None else tails.shape[-1],
        )
        scores = scores.reshape(-1, num_tails)
        return scores

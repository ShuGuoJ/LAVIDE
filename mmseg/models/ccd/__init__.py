from .bc_heads import BaseHeadBC, ConcatHead, CondConcathead, DWKConcatModule
from .encoder_decoder import EncoderDecoderCCD
from .map_encoders import BasicMapEncoder
from .ccd_heads import BaseHeadCCD, ClipHeadCCD, TripleHeadCCD, TripleHeadCCDDegradation
from .sem_heads import SegformerSemHead, SETRUPSemHead, UPerSemHead, FCNSemHead, SimpleSemHead
from .cross_modal.encoder_decoder import EncoderDecoderCMCD
from .cross_modal.bc_heads import CrossModalAttentionHead, CrossModalConcathead
from .cross_modal.sem_heads import CrossModalDummySemHead, CrossModalSegformerSemHead, WeaklySegformerSemHead

# clip
from .clip_head.encoder_decoder import EncoderDecoderClip
from .clip_head.bc_heads import CrossModalLavideHead
from .clip_head.enhanced_heads import ContextDecoder, MultiContextDecoder

# inst
from .inst_head import InstHeadBC

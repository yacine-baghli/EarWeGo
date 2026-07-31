"""
FAMILY E: joint bilateral + full-head model. Both ears of a subject and a low-resolution
full-head crop are encoded into ONE SUBJECT TOKEN that conditions a weight-shared per-ear
decoder. MODE=single|bilateral|bilateral_head is the ONLY difference between the three
arms, and the three arms have IDENTICAL parameter counts and identical state_dict keys.

WHAT IS BEING TESTED, AND WHAT WOULD FALSIFY IT. Two measured facts sit in tension.
(a) research/results/correlation_analyses.json: the per-ear oracle corrections correlate
    ACROSS THE TWO EARS OF A SUBJECT -- along-contour offset r 0.35..0.44, stretch
    r 0.39..0.52, across-contour r 0.45..0.56 -- against a GEOMETRY-MATCHED
    different-subject null of |r| <= 0.19. So a shared per-subject factor exists and is
    not explained by ear shape.
(b) research/results/context_probe.json: 121 hand-crafted global head / mesh-quality /
    crop / alignment / bilateral features predict those same corrections with max OOF
    R^2 = -0.0149 over 12 targets. Every one of them is "none".
This family asks the one question (b) leaves open: is the shared factor visible to a
LEARNED encoder of the raw geometry, when it can shape its own descriptor and is trained
against the landmark loss end to end rather than against a hand-picked scalar target?
The honest prior is NO -- (b) is a real negative result, the shipped 1.273 mm model
already pools its own ear globally, and six learned correction predictors returned ~zero
OOF R^2. The value of this family is therefore mostly in the CONTROL: if
bilateral == single to within the 0.0165 mm seed sd, the "use the other ear" idea is
closed with a like-for-like architecture instead of a feature-engineering proxy.

FALSIFICATION IS ONLY MEANINGFUL IF MODE IS THE ONLY CHANGE, so:
  * one context-set table (MEMBERS below) is read by NEEDS, by the augmenter and by the
    model, so a mode cannot disagree with itself;
  * the context encoder is applied to every member and pools them ORDER-INVARIANTLY, so
    the three modes differ ONLY in the CARDINALITY of the pooled set -- not one weight
    shape, not one layer, not one init;
  * the smoke test asserts equal parameter counts AND equal state_dict key sets across
    the three modes.
The cost of that choice is honest and stated: in MODE=single and MODE=bilateral the
`is_head` input column of the context encoder is never 1, so its weights exist and take
no gradient. That is the price of a bit-for-bit identical parameter tensor list, and it
is cheaper than a comparison confounded by a differently-shaped network.

THE SUBJECT TOKEN IS COMPUTED FROM OBSERVED GEOMETRY. NEVER FROM SUBJECT IDENTITY.
There is no nn.Embedding indexed by subject or by ear anywhere in this file (the only
embedding tables index the LANDMARK, 0..84). An identity-indexed table would be trained
to a value per training subject and would have NOTHING to look up for an unseen subject,
so it could not generalise; worse, during training it is a free per-subject parameter
fitted against that subject's own ground truth, which is leakage wearing an architecture
costume -- the network would appear to "use bilateral context" while actually memorising
the label. The token here is a function of point coordinates and normals only, and the
smoke test PROVES the absence of an identity path by running the same clouds under two
different `ear` indices and asserting the output is bit-identical.

THE TOKEN IS GENUINELY SHARED, AND THAT IS AN EXACT PROPERTY, NOT A HOPE.
The pooled set for ear e is
    {crop(e), crop(e^1)}                                        MODE=bilateral
    {crop(e), crop(e^1), head(e), head(e^1)}                    MODE=bilateral_head
where crop(e) is ear e's crop in ITS OWN canonical frame and head(e) is the subject's
head in ear e's canonical frame. That set is IDENTICAL for e and e^1, and the pool is
mean+max over members, so at inference the token of a subject's left ear equals the token
of its right ear to float32 rounding -- measured max abs difference 0.0 for the two-member
set and 7.5e-9 for the four-member one, where the only difference is the order the mean is
summed in. Two consequences:
  * the one-hot source channel marks EAR vs HEAD, never self vs partner. Marking
    self/partner would make the two ears see different sets and there would be no shared
    token at all -- this is the single easiest way to get this design wrong.
  * the crops are NOT brought into a common frame. Doing so would make ear e's set live in
    e's frame and ear e^1's set live in e^1's frame, i.e. two different sets, again no
    sharing. The token is therefore INTRINSIC to the two crop shapes and deliberately
    carries no ear-to-ear relative pose -- except through the two HEAD members, which are
    the same head in the two frames, so their difference IS that relative pose. That is
    the whole reason MODE=bilateral_head ships the head twice.
At TRAIN time the two ears of a subject are separate batch rows and get independent
augmentation draws, so their tokens differ by that augmentation. That is ordinary
augmentation noise, and it additionally trains the token to be pose-invariant.

HOW A BATCH PAIRS THE TWO EARS, AND WHAT HAPPENS IF A PARTNER IS MISSING.
train_family.py batches by EAR: one row = one TARGET ear, supervised on its own 85
landmarks, output (B, 85, 3). The partner arrives as ear-indexed CONTEXT columns that
research/code/build_bilateral_data.py has already permuted by e^1 (`pcp`, `pcp_nrm`,
`headp`, `headp_nrm`), because constraint 3 fixes subject = ear_index // 2 with the two
ears stored as consecutive (left, right) rows, so partner(e) = e ^ 1. The prep script
asserts that pairing against the pid/side arrays rather than assuming it.
  * A subject appears twice per epoch, once as target with its sibling as context and
    once the other way round. Nothing is double-counted: each row's loss reads only the
    target ear's ground truth.
  * NO COUPLING TERM IS POSSIBLE in this batching (the two ears are never in the same
    forward), and none is used. Because the decoder is weight-shared and the token is
    symmetric, decoding both ears jointly would be EXACTLY equivalent to two rows with
    the same token, so nothing is lost -- but a left/right consistency loss on the
    predictions is not expressible here and is deliberately left out.
  * MISSING PARTNER: `has_partner` (and `has_head`) are float 0/1 columns and the pool is
    MASKED. A member with validity 0 is dropped from the mean (denominator) and from the
    max (filled with -1e4); it is NOT replaced by a copy of the self member, which would
    double-count the target ear and bias the token towards it. With has_partner = 0 the
    bilateral token therefore equals, EXACTLY, the single-mode token with the same
    weights -- asserted in the smoke test. In this dataset every subject has both ears,
    so the mask is all ones; it exists so the model degrades gracefully at inference on a
    single-ear scan.

NO LEAKAGE (constraint 2). The context is OBSERVED GEOMETRY: point coordinates and
outward normals. Ground truth reaches loss() only, as train_family.py's separate `tg`
argument, and cannot enter a forward pass. The bilateral pathway also cannot smuggle a
training ear into a validation forward: both ears of a subject share a fold by
construction, so a validation ear's partner is always another validation ear.

FRAMES AND MIRRORING. Right ears are already mirrored into the left-ear canonical frame
upstream (diag(1,-1,1), then p -> (p - c0) @ R^T), and this family reuses that frame
unchanged for the crops. The HEAD clouds are produced in the same per-ear canonical frame
by build_bilateral_data.py, which flips the face winding and RECOMPUTES the vertex
normals rather than transforming the normal vector -- the outward normal under the
reflection M is M*n, and the winding-derived normal of the reflected mesh with UNCHANGED
winding is -(M*n), so "mirror the vector" silently ships inward normals. Measured on
P0001 and cross-checked against the shipped scratch/screen_data_2048nrm.npz: the
flip-and-recompute normals agree with the shipped ones at dot = +1.0000 and with -(M*n)
at dot = -1.0000. (NB the DOCSTRINGS of build_screen_extra.py and build_mesh_data.py both
say "n -> -(MIRROR*n)"; their CODE flips the winding and recomputes, which is the correct
route. The docstrings are stale -- do not copy them.)

AUGMENTATION IS OUR OWN, AND IT HAS TO BE. train_family.default_augment moves only
tensors shaped exactly like the cloud, so the (B, Nh, 3) head members would be left
unrotated and unscaled while the crop, the coarse init and the target are all rotated:
every shape check passes and the model trains against two inconsistent frames. See
bilateral_augment.

KNOWN LIMITS, STATED SO THEY ARE NOT DISCOVERED LATER.
  * MODE=single IS NOT THE SHIPPED BASELINE. It is the family-internal control: the same
    995,248-parameter network, with the context set reduced to the target ear's own crop.
    It therefore carries a global pathway the 813,232-parameter shipped model does not
    have, and it is redundant with the backbone's own global max-pool. Compare MODE arms
    to EACH OTHER; compare the family to the shipped 1.3144 mm pooled-OOF number only
    through MODE=single, and read any difference there as capacity, not as context.
  * The context members are subsampled by the SAME sub_frac as the decoder's cloud, so the
    head member is seen at 1920 points during training and 3072 at evaluation. That is the
    density mismatch the baseline already has for its own cloud (1280 vs 2048) and the pool
    is mean+max, which is largely density-robust -- but it is a mismatch, and CFG_SUB_FRAC
    moves it for every member at once.
  * THE PATHWAY STARTS NEAR-UNINFORMATIVE, AND THAT IS THE FIRST THING TO CHECK IF AN ARM
    COMES BACK EXACTLY NULL. Measured on the smoke test at initialisation: dropping the
    partner member changes the token by 0.35% of its norm and the PREDICTION by 6e-6 mm,
    because an untrained PointNet returns near-identical descriptors for two similar
    clouds. There is no optimisation barrier in that (the encoder still receives
    normal-magnitude gradients, and the baseline's own global max-pool starts equally
    uninformative), so no normalisation layer was added -- BatchNorm on the token would
    couple the batch and LayerNorm would strip the token's scale, and both would be a
    second change riding along with MODE. But it does mean a null result must be read with
    one extra diagnostic: load the trained checkpoint, compute the token over the
    validation ears and look at its across-SUBJECT standard deviation. If that is ~0 the
    arm never learned to use the context and the null is about optimisation, not about
    context. MODEL.forward returns the token as out["tok"] precisely so this is one line.
  * There is no left/right consistency term, no coupling loss, and no joint decode. See
    "HOW A BATCH PAIRS THE TWO EARS" above for why nothing is lost by that and what would
    be needed to add one.
  * The two head members carry the ear-to-ear relative pose only through the pipeline's own
    per-ear (R, c0). Those come from the coarse ear detection, so any systematic error in
    the alignment is INSIDE the head member's frame -- which is the point (the hypothesis
    is partly about alignment) but also means a null result cannot distinguish "no signal"
    from "the signal is exactly what the alignment already removed".

  MODE=bilateral python research/code/fam_bilateral.py          # CPU smoke test, all 3 modes
  MODE=bilateral_head python research/code/fam_bilateral.py
  FAMILY=bilateral MODE=bilateral FOLD=0 SEED=0 EPOCHS=1200 \
      DATA=scratch/bilat_data_2048.npz python3 research/code/train_family.py


ENVIRONMENT (all optional; every value is echoed into train_family.py's report)
------------------------------------------------------------------------------
  MODE      bilateral  single | bilateral | bilateral_head. MUST come from the ENVIRONMENT
                       and not CFG_MODE: train_family.py reads cls.NEEDS off the class
                       before instantiating it, so which context arrays are loaded is
                       fixed by then. Asserted against cfg["mode"].
  C         256   decoder width (the shipped baseline's 256)
  CTX_W     128   context-encoder width
  TOK       128   subject-token width
  NPASS     4     refinement passes            UNTIED  0   1 = untie the pass weights
  K         48    landmark window, points      GK      20  backbone graph neighbours
  DROPOUT   0.1   in the offset MLP            DEC_NRM 0   1 = feed normals to the DECODER
  HSCALE    100.0 mm, the head members' coordinate normaliser (crops use SCALE = 30)
  HJIT      0.93  mm/coord augmentation jitter for the HEAD members. The baseline's
                  0.25 mm/coord sits at 0.25/0.995 of the 2048-point crop spacing; the
                  head cloud's mean nearest-neighbour spacing, MEASURED over all 340 dev
                  ears of scratch/bilat_data_2048.npz, is 3.708 mm (3.411 .. 4.206), so
                  the same ratio is 0.25 * 3.708/0.995 = 0.932. Inheriting 0.25 would be
                  harmless but pointlessly weak; a jitter near 3.7 would erase the member.
  HEAD_SPACING_MM 3.708  the measured spacing HJIT is derived from and checked against.
                  build_bilateral_data.py stores the per-ear value as `head_spacing` and
                  prints the mean, so a rebuild at another HPTS/HRADIUS reports what to
                  set this to.
"""
import os, time
import numpy as np
import torch
import torch.nn as nn

NL, SCALE = 85, 30.0
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]

MODE = os.environ.get("MODE", "bilateral")            # single | bilateral | bilateral_head
C_WID = int(os.environ.get("C", "256"))
CTX_W = int(os.environ.get("CTX_W", "128"))
TOK = int(os.environ.get("TOK", "128"))
NPASS = int(os.environ.get("NPASS", "4"))
UNTIED = int(os.environ.get("UNTIED", "0"))
K = int(os.environ.get("K", "48"))
GK = int(os.environ.get("GK", "20"))
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
DEC_NRM = int(os.environ.get("DEC_NRM", "0"))
HSCALE = float(os.environ.get("HSCALE", "100.0"))
HEAD_SPACING_MM = float(os.environ.get("HEAD_SPACING_MM", "3.708"))
HJIT = float(os.environ.get("HJIT", round(0.25 * HEAD_SPACING_MM / 0.995, 4)))

MODES = ("single", "bilateral", "bilateral_head")

# ---------------------------------------------------------------------------------------
# THE ONE DEFINITION OF THE CONTEXT SET. cls.NEEDS, bilateral_augment and the model all
# read this table, so a mode cannot disagree with itself about which arrays exist, which
# ones are point-valued (rotate + scale + jitter) and which are direction-valued (rotate
# only). Adding a member is one row here and one array in build_bilateral_data.py.
#   points key, normals key, jitter cfg key, is_head, validity key, modes that include it
# `is_head` picks the coordinate normaliser AND the one-hot column, and it marks a KIND,
# never self-vs-partner -- see the module docstring.
MEMBERS = (
    ("pc",    "nrm",       "aug_jit", 0.0, None,          MODES),
    ("pcp",   "pcp_nrm",   "aug_jit", 0.0, "has_partner", MODES[1:]),
    ("head",  "head_nrm",  "hjit",    1.0, "has_head",    MODES[2:]),
    ("headp", "headp_nrm", "hjit",    1.0, "has_both",    MODES[2:]),
)
CTX_CIN = 3 + 3 + 2                     # xyz / scale, outward normal, (is_ear, is_head)
FLAGS = ("has_partner", "has_head")


def members_for(mode):
    assert mode in MODES, f"MODE={mode!r} is not one of {MODES}"
    return tuple(m for m in MEMBERS if mode in m[5])


def needs_for(mode):
    """The DATA keys this mode pulls out of the npz. `pc` is the batch's own cloud, so it
    is never requested; everything else a member reads is."""
    ks, fl = [], set()
    for pk, nk, _, _, vk, _ in members_for(mode):
        ks += ([] if pk == "pc" else [pk]) + [nk]
        fl |= set(FLAGS) if vk == "has_both" else ({vk} if vk else set())
    return tuple(ks) + tuple(sorted(fl))


def rotates_for(mode):
    return tuple(m[1] for m in members_for(mode))


# --------------------------------------------------------------- augmentation
def rand_rot(B, maxang, gen, dev):
    """train_family.rand_rot verbatim, so the two augmenters draw the same distribution."""
    ax = torch.randn(B, 3, device=dev, generator=gen)
    ax = ax / ax.norm(dim=1, keepdim=True)
    ang = (torch.rand(B, device=dev, generator=gen) - .5) * maxang
    c, s = ang.cos(), ang.sin(); x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]; C = 1 - c
    return torch.stack([
        torch.stack([c + x*x*C, x*y*C - z*s, x*z*C + y*s], -1),
        torch.stack([y*x*C + z*s, c + y*y*C, y*z*C - x*s], -1),
        torch.stack([z*x*C - y*s, z*y*C + x*s, c + z*z*C], -1)], 1)


def _rot(t, R):
    """row vectors: t @ R^T, per batch item, for any number of middle axes."""
    return torch.einsum("b...j,bij->b...i", t, R)


def bilateral_augment(b, tg, cfg, rotates, gen):
    """ONE per-ear similarity applied to the target cloud, the coarse init, the target AND
    every context member; point members additionally get jitter, normals get the rotation
    only.

    WHY NOT default_augment. It moves a tensor only when its shape is exactly the cloud's
    (B,S,N,C). The head members are (B,Nh,3) with Nh != N, so they would be left
    completely alone -- unrotated and unscaled -- while pc/coarse/tg are rotated and
    scaled. Every shape check passes and the family trains against two inconsistent
    frames. cls.ROTATES cannot rescue it either: it rotates without scaling, which is
    right for a normal and wrong for a point cloud.

    WHY ONE TRANSFORM FOR ALL MEMBERS RATHER THAN ONE EACH. The two head members are the
    SAME head expressed in the two ears' canonical frames, so the difference between them
    IS the ear-to-ear relative pose; independent per-member rotations would destroy it.
    For the two crops, each already living in its own canonical frame, a shared rotation
    is not a claim about their relative pose -- it just keeps one similarity per row.

    Every member is subsampled to sub_frac of its own point count with its OWN index
    draw, which is why the jitter is per-member (cfg['aug_jit'] for the crops,
    cfg['hjit'] for the head) rather than one global millimetre number.
    """
    pc = b["pc"]; B, dev = pc.shape[0], pc.device
    R = rand_rot(B, cfg["aug_rot"], gen, dev)
    sc = 1 + (torch.rand(B, 1, 1, device=dev, generator=gen) - .5) * cfg["aug_scale"]
    out = dict(b)
    seen = {"coarse", "ear"}
    for pk, nk, jk, _, vk, _ in MEMBERS:
        if pk not in b:
            continue
        assert nk in b, (f"context member '{pk}' is in the batch but its normals '{nk}' are "
                         f"not; MEMBERS pairs them and cls.NEEDS must request both")
        p, n = b[pk], b[nk]
        assert p.shape == n.shape, f"{pk} {tuple(p.shape)} != {nk} {tuple(n.shape)}"
        N = p.shape[-2]
        nsub = max(8, min(N, int(round(N * cfg["sub_frac"]))))
        sub = torch.rand(p.shape[:-1], device=dev, generator=gen).argsort(-1)[..., :nsub]
        g = sub[..., None].expand(*sub.shape, 3)
        p, n = torch.gather(p, -2, g), torch.gather(n, -2, g)
        s_ = sc[:, None] if p.dim() == 4 else sc
        out[pk] = _rot(p, R) * s_ + \
            torch.randn(p.shape, device=dev, generator=gen) * float(cfg[jk])
        out[nk] = _rot(n, R)                        # a direction: rotate, never scale
        seen |= {pk, nk}
    seen |= set(FLAGS)
    stray = [k for k, v in b.items() if torch.is_tensor(v) and k not in seen]
    assert not stray, (f"bilateral_augment does not know how to move {stray}. Every tensor in "
                       f"the batch must be a point member, a normals member, `coarse`, `ear` "
                       f"or a {FLAGS} flag -- an unclassified key would stay in the ORIGINAL "
                       f"frame while everything else is rotated. Add it to MEMBERS.")
    assert set(rotates) <= seen, f"cls.ROTATES names {sorted(set(rotates) - seen)}, not in the batch"
    out["coarse"] = _rot(b["coarse"], R) * sc + \
        torch.randn(b["coarse"].shape, device=dev, generator=gen) * cfg["aug_qjit"]
    return out, _rot(tg, R) * sc


# --------------------------------------------------------------- context / subject token
class CtxEncoder(nn.Module):
    """PointNet-style set encoder: one GLOBAL descriptor per context member.

    Deliberately neighbourhood-free. The per-ear decoder already sees local geometry
    through its own kNN graph; what this pathway adds, and the only thing it is allowed to
    add, is a global descriptor of a piece of observed geometry. Making it a graph network
    would (a) cost a 3072x3072 cdist per member per row and (b) turn a null result into an
    ambiguous one, because a null could then be blamed on the extra capacity rather than
    on the absence of signal. One round of global feedback (max-pool concatenated back to
    every point) is the standard way to let per-point features be computed in the context
    of the whole member, and is the entire deviation from a plain PointNet.
    """
    def __init__(self, cin, w):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(cin, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())
        self.b = nn.Sequential(nn.Linear(2 * w, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())

    def forward(self, x):
        h = self.a(x)
        h = self.b(torch.cat([h, h.max(1, keepdim=True).values.expand_as(h)], -1))
        return torch.cat([h.mean(1), h.max(1).values], -1)              # (B, 2w)


class SubjectToken(nn.Module):
    """The shared subject token: a MASKED, ORDER-INVARIANT pool over the context set.

    NOT AN EMBEDDING TABLE. There is no nn.Parameter here indexed by subject or by ear,
    and no argument through which an identity could arrive -- `forward` takes point
    coordinates and normals. An identity-indexed table would have nothing to look up for
    an unseen subject and, during training, would be a free per-subject parameter fitted
    against that subject's own labels, i.e. leakage in the shape of an architecture.

    Order-invariance is what makes the token SHARED: the pooled set is the same set for
    both ears of the subject, so both get the same token. mean and max are the two
    symmetric pools that survive a varying member count unchanged, which is also why the
    same weights serve all three MODEs.
    """
    def __init__(self, cin, w, tok, hscale):
        super().__init__()
        self.enc = CtxEncoder(cin, w)
        self.mlp = nn.Sequential(nn.Linear(4 * w, tok), nn.ReLU(),
                                 nn.Linear(tok, tok), nn.ReLU())
        self.hscale = hscale

    def forward(self, mem):
        """mem: list of (pts (B,P,3), nrm (B,P,3), is_head float, valid (B,)) -> (B, tok)"""
        assert mem, "the context set is empty"
        F, V = [], []
        for pts, nrm, is_head, ok in mem:
            oh = pts.new_tensor([1.0 - is_head, is_head]).expand(*pts.shape[:2], 2)
            F.append(self.enc(torch.cat([pts / (self.hscale if is_head else SCALE),
                                         nrm, oh], -1)))
            V.append(ok)
        F = torch.stack(F, 1)                                          # (B, m, 2w)
        v = torch.stack(V, 1).clamp(0.0, 1.0)[..., None]               # (B, m, 1)
        # the self member is valid by construction, so the denominator is never 0 and the
        # max is never all -1e4; asserted rather than clamped away silently
        assert bool((v.sum(1) > 0).all()), "an ear has no valid context member at all"
        mean = (F * v).sum(1) / v.sum(1)
        return self.mlp(torch.cat([mean, F.masked_fill(v == 0, -1e4).max(1).values], -1))


# --------------------------------------------------------------- per-ear decoder
def knn(q, pc, k):
    return torch.cdist(q, pc).topk(k, largest=False, dim=-1).indices


def edge_conv(gidx, feat, mlp):
    B, P, C = feat.shape; k = gidx.shape[-1]
    fj = torch.gather(feat, 1, gidx.reshape(B, P * k, 1).expand(-1, -1, C)).view(B, P, k, C)
    return mlp(torch.cat([feat[:, :, None, :].expand(-1, -1, k, -1), fj - feat[:, :, None, :]],
                         -1)).max(2).values


class Head(nn.Module):
    """one refinement pass: bounded offset, then a softmax snap onto the gathered surface
    points. gpu_screen.py's Head, with K / dropout as attributes instead of module globals
    so the three MODE builds cannot disagree about them. The two embedding tables index
    the LANDMARK (0..84) and nothing else."""
    def __init__(self, C, k, dropout, max_off=None):
        super().__init__()
        self.emb, self.embO = nn.Embedding(NL, 32), nn.Embedding(NL, 32)
        self.offset = nn.Sequential(nn.Linear(2 * C + 32, 256), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.attn = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.C, self.k, self.max_off = C, k, max_off

    def gather(self, pc, h, q):
        B, k = pc.shape[0], min(self.k, pc.shape[1])
        idx = knn(q, pc, k).reshape(B, NL * k)
        return (torch.gather(h, 1, idx[..., None].expand(-1, -1, self.C)).view(B, NL, k, self.C),
                torch.gather(pc, 1, idx[..., None].expand(-1, -1, 3)).view(B, NL, k, 3))

    def forward(self, pc, h, q, diag=False):
        fK, _ = self.gather(pc, h, q)
        ar = torch.arange(NL, device=pc.device)
        ctx = torch.cat([fK.mean(2), fK.max(2).values], -1)
        off = self.offset(torch.cat([ctx, self.embO(ar)[None].expand(pc.shape[0], -1, -1)], -1))
        if self.max_off is not None:
            off = self.max_off * torch.tanh(off / max(self.max_off, 1e-6))
        q1 = q + off
        fK2, pK2 = self.gather(pc, h, q1)
        rel = (pK2 - q1[:, :, None, :]) / SCALE
        e = self.emb(ar)[None, :, None, :].expand(pc.shape[0], NL, pK2.shape[2], 32)
        w = torch.softmax(self.attn(torch.cat([fK2, rel, e], -1)).squeeze(-1), -1)
        q2 = (w[..., None] * pK2).sum(2)
        return q1, q2, ({"w": w.detach(), "pK": pK2.detach()} if diag else None)


class EarDecoder(nn.Module):
    """The shipped 1.273 mm architecture (DGCNN backbone -> NPASS offset/snap passes ->
    per-contour smoothing), with the subject token as extra input to ONE layer.

    ONE decoder, weight-shared, applied to whichever ear is the target -- "both ear
    decoders" in the sense that both ears are decoded by it. Right ears are already
    mirrored into the left-ear canonical frame upstream, so a separate right-ear decoder
    would only add a left/right capacity asymmetry and halve the ears each copy sees.

    THE TOKEN'S ONLY INJECTION POINT is the `mix` layer, which the baseline already uses
    to fuse per-point features with the ear's own global max-pool. Concatenating the token
    there gives every downstream stage (all passes, the contour nets) access to it through
    h, keeps the extra parameters to one Linear's extra input columns, and keeps that
    Linear's shape IDENTICAL in all three modes.
    """
    def __init__(self, C, cin, tok, npass, untied, k, gk, dropout):
        super().__init__()
        self.ec1 = nn.Sequential(nn.Linear(2 * cin, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ec2 = nn.Sequential(nn.Linear(2 * 64, 128), nn.ReLU())
        self.ec3 = nn.Sequential(nn.Linear(2 * 128, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(320, C), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(2 * C + tok, C), nn.ReLU())
        # max_off stays None: the shipped 1.273 mm `base` variant has an UNBOUNDED offset
        # (gpu_screen.py sets OFFS only for untied6). Head keeps the tanh branch so a
        # bounded-offset variant is one DEFAULTS entry, but this family does not sweep it --
        # the offset ladder is a different experiment and would ride along with MODE.
        self.heads = nn.ModuleList([Head(C, k, dropout)
                                    for _ in range(npass if untied else 1)])
        self.lmfeat = nn.Sequential(nn.Linear(C, 64), nn.ReLU())
        self.embC = nn.Embedding(NL, 32)
        self.cnet = nn.ModuleList([
            nn.Sequential(nn.Conv1d(3 + 64 + 32, 96, 5, padding=2), nn.ReLU(),
                          nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 3, 1))
            for _ in CONTOURS])
        self.C, self.npass, self.untied, self.gk = C, npass, untied, gk

    def backbone(self, pc, tok, ft=None):
        pos = pc / SCALE
        gidx = knn(pos, pos, min(self.gk, pc.shape[1]))
        x = pos if ft is None else torch.cat([pos, ft], -1)
        h1 = edge_conv(gidx, x, self.ec1)
        h2 = edge_conv(gidx, h1, self.ec2)
        h3 = edge_conv(gidx, h2, self.ec3)
        h = self.fuse(torch.cat([h1, h2, h3], -1))
        g = h.max(1, keepdim=True).values.expand(-1, pc.shape[1], -1)
        return self.mix(torch.cat([h, g, tok[:, None, :].expand(-1, pc.shape[1], -1)], -1))

    def contour(self, pc, h, q):
        B = pc.shape[0]
        idx = knn(q, pc, 1).squeeze(-1)
        f = self.lmfeat(torch.gather(h, 1, idx[..., None].expand(-1, -1, self.C)))
        e = self.embC(torch.arange(NL, device=pc.device))[None].expand(B, -1, -1)
        inp = torch.cat([q / SCALE, f, e], -1)
        out = torch.zeros(B, NL, 3, device=pc.device, dtype=q.dtype)
        for (lo, hi), net in zip(CONTOURS, self.cnet):
            out[:, lo:hi + 1] = net(inp[:, lo:hi + 1].transpose(1, 2)).transpose(1, 2)
        return q + out

    def forward(self, pc, q0, tok, ft=None, diag=False):
        h = self.backbone(pc, tok, ft)
        outs, info, q = [], [], q0
        for i in range(self.npass):
            q1, q2, dg = self.heads[i if self.untied else 0](pc, h, q, diag)
            outs.append((q1, q2)); info.append(dg); q = q2
        return outs, self.contour(pc, h, q), info


class BilateralNet(nn.Module):
    """SubjectToken(context set) -> EarDecoder(target ear). MODE only changes the set."""
    def __init__(self, mode=MODE, C=C_WID, ctx_w=CTX_W, tok=TOK, npass=NPASS,
                 untied=UNTIED, k=K, gk=GK, dropout=DROPOUT, dec_nrm=DEC_NRM,
                 hscale=HSCALE):
        super().__init__()
        self.mode, self.dec_nrm = mode, bool(dec_nrm)
        self.mem = members_for(mode)
        self.token = SubjectToken(CTX_CIN, ctx_w, tok, hscale)
        self.dec = EarDecoder(C, 3 + 3 * bool(dec_nrm), tok, npass, bool(untied), k, gk, dropout)

    def members(self, b):
        one = b["pc"].new_ones(b["pc"].shape[0])
        out = []
        for pk, nk, _, is_head, vk, _ in self.mem:
            assert pk in b and nk in b, \
                (f"MODE={self.mode} needs '{pk}' and '{nk}' in the batch; cls.NEEDS is "
                 f"{needs_for(self.mode)} and DATA must carry all of them")
            ok = one if vk is None else \
                (b["has_head"] * b["has_partner"] if vk == "has_both" else b[vk])
            out.append((b[pk], b[nk], is_head, ok))
        return out

    def forward(self, b, diag=False):
        assert "true" not in b, "ground truth reached a forward pass (constraint 2)"
        t = self.token(self.members(b))
        return self.dec(b["pc"], b["coarse"], t,
                        b["nrm"] if self.dec_nrm else None, diag) + (t,)


# --------------------------------------------------------------- train_family adapter
class MODEL(nn.Module):
    """Family E as train_family.py's REGISTRY["bilateral"] resolves it."""
    DEFAULTS = dict(mode=MODE, width=C_WID, ctx_w=CTX_W, tok=TOK, npass=NPASS,
                    untied=UNTIED, k=K, gk=GK, dropout=DROPOUT, dec_nrm=DEC_NRM,
                    hscale=HSCALE, hjit=HJIT)
    # MODE is deliberately ABSENT: it is the experiment, not a hyperparameter, and it
    # cannot be swept anyway because cls.NEEDS is read off the class before it is built.
    SEARCH_SPACE = dict(ctx_w=[64, 128, 256], tok=[64, 128, 256], width=[128, 256],
                        npass=[4, 6], untied=[0, 1], k=[32, 48, 96], gk=[12, 20, 32],
                        dropout=[0.0, 0.1, 0.2], dec_nrm=[0, 1],
                        hjit=[round(f * HJIT, 4) for f in (0.5, 1.0, 2.0)],
                        lr=[7e-4, 1.5e-3, 3e-3], sub_frac=[0.5, 0.625, 0.8])
    NEEDS = needs_for(MODE)
    ROTATES = rotates_for(MODE)
    SAMPLES = 1
    AUGMENT = staticmethod(bilateral_augment)

    def __init__(self, cfg, meta):
        super().__init__()
        assert str(cfg["mode"]) == MODE, (
            f"cfg mode={cfg['mode']!r} but the ENVIRONMENT has MODE={MODE!r}. Set MODE in "
            f"the environment, not CFG_MODE: train_family.py reads cls.NEEDS "
            f"({needs_for(MODE)}) before instantiating, so which context arrays are loaded "
            f"is already fixed by the time this cfg is seen.")
        assert float(cfg["hjit"]) <= HEAD_SPACING_MM, (
            f"hjit={cfg['hjit']} mm/coord exceeds the {HEAD_SPACING_MM} mm measured mean "
            f"nearest-neighbour spacing of the head cloud, so the augmenter displaces every "
            f"head point past its neighbour and erases the member -- at TRAIN time only, "
            f"which is a train/test mismatch that would return null for a reason having "
            f"nothing to do with head context. Check the spacing this DATA actually has "
            f"(build_bilateral_data.py prints and stores head_spacing) and set "
            f"HEAD_SPACING_MM if it differs.")
        self.net = BilateralNet(MODE, int(cfg["width"]), int(cfg["ctx_w"]), int(cfg["tok"]),
                                int(cfg["npass"]), int(cfg["untied"]), int(cfg["k"]),
                                int(cfg["gk"]), float(cfg["dropout"]), int(cfg["dec_nrm"]),
                                float(cfg["hscale"]))

    def forward(self, b):
        outs, final, _, tok = self.net(b)
        return {"pred": final, "aux": [q for pair in outs for q in pair], "tok": tok}

    def loss(self, out, tg):
        """gpu_screen.py's schedule verbatim: within a pass the pre-snap offset gets 0.4 and
        the snapped point 1.0; across passes the weight halves backwards from the last."""
        a = out["aux"]; n = len(a) // 2
        w = [0.5 ** (n - 1 - t) for t in range(n)]; s = sum(w)
        L = sum((w[t] / s) * (0.4 * ((a[2 * t] - tg) ** 2).sum(-1).mean()
                              + ((a[2 * t + 1] - tg) ** 2).sum(-1).mean()) for t in range(n))
        return L + ((out["pred"] - tg) ** 2).sum(-1).mean()


# --------------------------------------------------------------- smoke test
def _subject(gen, k, npts, hpts):
    """one synthetic subject: two ear-sized crops in their OWN canonical frames, plus the
    same head-sized shell expressed in the two frames (a reflection + rotation apart, as
    the real left/right pair is). Returns (crops, crop normals, heads, head normals)."""
    def shell(n, ax, ph):
        v = torch.randn(n, 3, generator=gen)
        v = v / v.norm(dim=1, keepdim=True)
        p = v * torch.tensor(ax) + torch.tensor([0.0, 0.0, ph])
        return p, v
    # the two ears of a subject must be DIFFERENT shapes, or the single-mode token is
    # accidentally shared too and the sharing test proves nothing
    crops, cn = zip(*[shell(npts, [30.0 + 8.0 * j + k, 22.0 - 6.0 * j, 14.0 + 4.0 * j], 3.0 * j)
                      for j in (0, 1)])
    H, HN = shell(hpts, [95.0, 70.0, 80.0], 3.0 * k)
    M = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
    A = rand_rot(1, 2.0, gen, "cpu")[0]
    # the partner's view of the same head: a reflection composed with a rotation. For a
    # det=-1 map the outward normal is M*n (NOT -(M*n)) -- the prep gets this by flipping
    # the winding and recomputing; here it is applied directly, which is the same thing.
    T = M @ A
    return list(crops), list(cn), [H, H @ T.T], [HN, HN @ T.T]


def _bundle(ne=8, npts=256, hpts=192, seed=0):
    """ear-indexed arrays exactly as build_bilateral_data.py emits them, for ne//2
    subjects. pcp/headp are the e^1 permutation, which is the contract under test."""
    gen = torch.Generator().manual_seed(seed)
    pc, nrm, hd, hn = [], [], [], []
    for k in range(ne // 2):
        c, cn, h, hnn = _subject(gen, k, npts, hpts)
        pc += c; nrm += cn; hd += h; hn += hnn
    pc, nrm, hd, hn = (torch.stack(x) for x in (pc, nrm, hd, hn))
    sw = torch.arange(ne) ^ 1
    ring = torch.stack([30 * torch.linspace(0, 6.28, NL).cos(),
                        22 * torch.linspace(0, 6.28, NL).sin(),
                        torch.zeros(NL)], -1)
    q0 = ring[None].expand(ne, -1, -1) + torch.randn(ne, NL, 3, generator=gen) * 1.5
    return dict(pc=pc, nrm=nrm, pcp=pc[sw], pcp_nrm=nrm[sw], head=hd, head_nrm=hn,
                headp=hd[sw], headp_nrm=hn[sw], coarse=q0,
                has_partner=torch.ones(ne), has_head=torch.ones(ne),
                ear=torch.arange(ne))


def _batch(d, rows, mode, **over):
    b = {k: (v[rows] if torch.is_tensor(v) and v.shape[0] == d["pc"].shape[0] else v)
         for k, v in d.items()}
    b = {k: v for k, v in b.items() if k in ("pc", "coarse", "ear") + needs_for(mode) or
         k in FLAGS}
    b.update(over)
    return b


def smoke():
    t0 = time.time()
    print("=" * 78)
    print(f"FAMILY E  MODE={MODE}  C={C_WID} CTX_W={CTX_W} TOK={TOK} NPASS={NPASS} "
          f"UNTIED={UNTIED} K={K} GK={GK} DEC_NRM={DEC_NRM} HSCALE={HSCALE} HJIT={HJIT}")
    torch.manual_seed(0); np.random.seed(0)
    NE, NPTS, HPTS = 8, 256, 192
    d = _bundle(NE, NPTS, HPTS)
    tg = d["coarse"] + 0.4
    print(f"synthetic bundle: {NE} ears / {NE//2} subjects, crop {tuple(d['pc'].shape[1:])} "
          f"head {tuple(d['head'].shape[1:])}, partner = ear^1")

    # ---- 1. every mode: forward, backward, shapes, params -----------------------------
    nets, npars, keys = {}, {}, {}
    for mode in MODES:
        torch.manual_seed(0)                       # same init stream in all three modes
        net = BilateralNet(mode)
        b = _batch(d, [0, 1], mode)
        outs, final, _, tok = net(b, diag=True)
        L = ((final - tg[[0, 1]]) ** 2).sum(-1).mean() + \
            sum(((q - tg[[0, 1]]) ** 2).sum(-1).mean() for p in outs for q in p)
        L.backward()
        gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
        npars[mode] = sum(p.numel() for p in net.parameters())
        keys[mode] = tuple(sorted(net.state_dict()))
        nets[mode] = net
        print(f"  [{mode:15s}] members {len(net.mem)}  params {npars[mode]:,}  "
              f"pred {tuple(final.shape)}  aux {2*len(outs)}x{tuple(outs[0][0].shape)}  "
              f"tok {tuple(tok.shape)}  loss {float(L):.3f}  grad-norm {gn:.2f}")
        assert final.shape == (2, NL, 3), final.shape
        assert torch.isfinite(final).all() and gn > 0
    assert len(set(npars.values())) == 1, f"parameter counts differ across modes: {npars}"
    assert len(set(keys.values())) == 1, "state_dict keys differ across modes"
    print(f"  ALL THREE MODES: {npars[MODES[0]]:,} params and identical state_dict keys "
          f"({len(keys[MODES[0]])} tensors) -- MODE changes only the pooled set's size")

    # ---- 2. the token really is SHARED between the two ears of a subject -------------
    def tokens(mode, net, rows, **over):
        with torch.no_grad():
            return net.token(net.members(_batch(d, rows, mode, **over)))
    for mode in MODES:
        tl = tokens(mode, nets[mode], [0, 2, 4, 6])          # left ears
        tr = tokens(mode, nets[mode], [1, 3, 5, 7])          # their partners
        dv = float((tl - tr).abs().max())
        if mode == "single":
            # an UNTRAINED PointNet returns similar descriptors for similar clouds, so the
            # bar here is only "ear-specific at all"; the meaningful number is the exact
            # zero the other two modes have to hit
            assert dv > 1e-5, "single-mode tokens coincide -- the ears are not distinct"
            print(f"  [{mode:15s}] token(left) vs token(right) max|diff| {dv:.2e}  "
                  f"(EAR-SPECIFIC, as it must be: the set is {{own crop}})")
        else:
            assert dv < 1e-6, f"{mode}: the subject token is not shared (max diff {dv:.2e})"
            print(f"  [{mode:15s}] token(left) vs token(right) max|diff| {dv:.2e}  "
                  f"(SHARED -- the set is the same set for both ears)")

    # ---- 3. no identity path: the `ear` index cannot change anything -----------------
    bm = nets["bilateral_head"].eval()
    with torch.no_grad():
        o1 = bm(_batch(d, [0, 1], "bilateral_head"))[1]
        o2 = bm(_batch(d, [0, 1], "bilateral_head",
                       ear=torch.tensor([317, 4])))[1]
    assert float((o1 - o2).abs().max()) == 0.0, "the ear index changed the output -- an " \
        "identity-indexed parameter exists somewhere"
    print(f"  ear index 0,1 -> 317,4 changes the prediction by exactly "
          f"{float((o1-o2).abs().max()):.1f} mm: no identity is read anywhere")

    # ---- 4. a missing partner degrades to the single-ear token, EXACTLY --------------
    sm = BilateralNet("single")
    sm.load_state_dict(nets["bilateral"].state_dict())       # identical key sets, so this works
    sm.eval()
    with torch.no_grad():
        t_no = tokens("bilateral", nets["bilateral"].eval(), [0, 1],
                      has_partner=torch.zeros(2))
        t_single = tokens("single", sm, [0, 1])
        t_yes = tokens("bilateral", nets["bilateral"], [0, 1])
    assert float((t_no - t_single).abs().max()) < 1e-6, \
        "has_partner=0 does not reproduce the single-ear token -- the mask is not working"
    print(f"  has_partner=0 -> token equals the single-member token to "
          f"{float((t_no-t_single).abs().max()):.2e}; with the partner it moves "
          f"{float((t_yes-t_single).norm(dim=-1).mean()):.4f} (L2, so the partner is used)")
    # the context pathway must actually REACH the prediction, or all three arms would be
    # the same experiment. At init the effect is tiny; the point is that it is not zero.
    with torch.no_grad():
        p1 = nets["bilateral"](_batch(d, [0, 1], "bilateral"))[1]
        p0 = nets["bilateral"](_batch(d, [0, 1], "bilateral",
                                      has_partner=torch.zeros(2)))[1]
    assert float((p1 - p0).abs().max()) > 0, \
        "dropping the partner does not change the prediction -- the token is not wired in"
    print(f"  ... and dropping it moves the PREDICTION by "
          f"{float((p1-p0).norm(dim=-1).mean()):.2e} mm mean / "
          f"{float((p1-p0).norm(dim=-1).max()):.2e} mm max (untrained, so small; the "
          f"assertion is only that the pathway reaches the output at all)")

    # ---- 5. the augmentation is ONE per-ear similarity over EVERY member -------------
    import train_family as T
    acfg = {**T.TRAIN_DEFAULTS, **MODEL.DEFAULTS, "aug_jit": 0.0, "hjit": 0.0,
            "aug_qjit": 0.0, "sub_frac": 1.0}
    g = torch.Generator().manual_seed(3)
    b0 = _batch(d, [0, 1], "bilateral_head")
    b0 = {k: (v[:, None] if k in ("pc", "nrm", "pcp", "pcp_nrm") else v)
          for k, v in b0.items()}                            # (B,S,N,3) as the trainer feeds it
    b1, tg1 = bilateral_augment(b0, tg[[0, 1]], acfg, MODEL.ROTATES, g)

    def spec(t):
        t = t.double().reshape(t.shape[0], -1, 3)
        return (t[:, :, None] - t[:, None]).norm(dim=-1).flatten(1).sort(-1).values

    scales = {}
    for nm in ("pc", "pcp", "head", "headp", "coarse"):
        s0, s1 = spec(b0[nm]), spec(b1[nm])
        sc = s1.sum(-1) / s0.sum(-1)
        assert float((s1 - sc[:, None] * s0).abs().max()) < 1e-4, \
            f"'{nm}' is not moved by a per-ear similarity"
        scales[nm] = sc
    s0, s1 = spec(tg[[0, 1]]), spec(tg1)
    scales["tg"] = s1.sum(-1) / s0.sum(-1)
    for nm, sc in scales.items():
        assert torch.allclose(sc, scales["tg"], atol=1e-6), \
            f"'{nm}' scale {sc.tolist()} != target scale {scales['tg'].tolist()}"
    for nm in ("nrm", "pcp_nrm", "head_nrm", "headp_nrm"):
        n1 = b1[nm].reshape(-1, 3).norm(dim=-1)
        assert float((n1 - 1).abs().max()) < 1e-5, f"'{nm}' was scaled, not just rotated"
    print(f"  augmentation: ONE similarity per ear across pc/pcp/head/headp/coarse/target, "
          f"shared scale {[round(float(x),6) for x in scales['tg']]}; all four normal "
          f"fields stayed unit-norm")
    try:
        bilateral_augment({**b0, "mystery": torch.zeros(2, 4)}, tg[[0, 1]], acfg,
                          MODEL.ROTATES, g)
        raise SystemExit("an unclassified batch key was silently left in the old frame")
    except AssertionError as e:
        assert "does not know how to move" in str(e), str(e)
    print("  refused: an unclassified tensor in the batch (it would stay unrotated)")

    # ---- 6. the MODEL adapter and the two guards ------------------------------------
    meta = dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=NPTS, fold=0, dev="cpu",
                n_train_ears=NE, artefacts={})
    cfg = {**T.TRAIN_DEFAULTS, **MODEL.DEFAULTS}
    m = MODEL(cfg, meta)
    b = _batch(d, [0, 1], MODE)
    o = m(b)
    L = T.default_loss(o, tg[[0, 1]], m, b)
    L.backward()
    gn = sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
    assert o["pred"].shape == (2, NL, 3), o["pred"].shape
    assert len(o["aux"]) == 2 * NPASS and gn > 0
    print(f"  MODEL adapter (MODE={MODE}): pred {tuple(o['pred'].shape)} aux {len(o['aux'])} "
          f"params {sum(p.numel() for p in m.parameters()):,} loss {float(L):.3f} "
          f"grad-norm {gn:.2f}  NEEDS {MODEL.NEEDS}")
    for bad, want in ((dict(mode="nonsense"), "ENVIRONMENT has MODE"),
                      (dict(hjit=HEAD_SPACING_MM * 1.01), "nearest-neighbour spacing")):
        try:
            MODEL({**cfg, **bad}, meta)
            raise SystemExit(f"MODEL accepted {bad} -- the '{want}' guard is dead")
        except AssertionError as e:
            assert want in str(e), f"wrong guard fired for {bad}: {e}"
    print(f"  refused: cfg mode != env MODE | hjit > the {HEAD_SPACING_MM} mm head spacing")

    # ---- 7. the trainer end to end, in every mode, incl. the full pipeline ----------
    # reload() rebuilds MODE / MEMBERS / MODEL.NEEDS from the environment. It hits the
    # SEPARATE copy of this module that import_module creates under the name
    # "fam_bilateral" (this one is "__main__"), so nothing in the running smoke test is
    # rebound -- and that second copy is exactly the one resolve_family() will import.
    import tempfile, importlib
    tmp = os.environ.get("SMOKE_DIR", os.path.join(tempfile.gettempdir(), "fam_bilateral_smoke"))
    dp, tp, sp, _ = T.fake_bundle(tmp, ne=20, npts=NPTS)
    z = dict(np.load(dp, allow_pickle=True))
    sw = np.arange(20) ^ 1
    rs = np.random.RandomState(0)
    nr = z["clouds"] / np.linalg.norm(z["clouds"], axis=-1, keepdims=True).clip(1e-9)
    hd = (rs.randn(20, HPTS, 3) * np.float32([60., 45., 50.])).astype(np.float32)
    hn = (hd / np.linalg.norm(hd, axis=-1, keepdims=True)).astype(np.float32)
    z.update(nrm=nr.astype(np.float32), pcp=z["clouds"][sw], pcp_nrm=nr[sw].astype(np.float32),
             head=hd, head_nrm=hn, headp=hd[sw], headp_nrm=hn[sw],
             has_partner=np.ones(20, np.float32), has_head=np.ones(20, np.float32))
    dp2 = f"{tmp}/bilat_fake.npz"
    np.savez(dp2, **z)
    keep = {k: os.environ.get(k) for k in
            ("FAMILY", "FAMILY_MODULE", "FOLD", "SEED", "EPOCHS", "WORK", "DATA", "TRIS",
             "SSM", "TTA", "EVAL_EVERY", "ALIAS", "CFG_BS", "TAG", "VARIANT", "MODE")}
    res = {}
    for mode in MODES:
        os.environ.update(FAMILY="bilateral", FAMILY_MODULE="fam_bilateral", FOLD="0",
                          SEED="0", EPOCHS="2", WORK=tmp, DATA=dp2, TRIS=tp, SSM=sp,
                          TTA="1", EVAL_EVERY="2", ALIAS="0", CFG_BS="4", MODE=mode,
                          VARIANT=f"bilateral_{mode}", TAG=f"fam_bilateral_{mode}")
        fb = importlib.reload(importlib.import_module("fam_bilateral"))
        assert fb.MODE == mode and fb.MODEL.NEEDS == fb.needs_for(mode)
        res[mode] = T.main()
        print(f"  [{mode:15s}] trainer OK: raw {res[mode]['ordered_MLE_mm']:.4f} -> full "
              f"{res[mode]['ordered_MLE_full_mm']:.4f} mm | params {res[mode]['params']:,} "
              f"| needs {res[mode]['config']['_needs']} | aug "
              f"{res[mode]['config']['_augment']}")
    for k, v in keep.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    assert len({r["params"] for r in res.values()}) == 1, \
        f"the trainer built different sizes per mode: {[r['params'] for r in res.values()]}"
    assert all(r["config"]["_augment"] == "bilateral_augment" for r in res.values())
    assert all(r["ordered_MLE_full_mm"] is not None for r in res.values())
    print(f"  the trainer built {res['single']['params']:,} params in all three modes; "
          f"full pipeline ran in all three")
    print(f"SMOKE PASS ({time.time()-t0:.0f}s)")
    print("=" * 78)


if __name__ == "__main__":
    smoke()

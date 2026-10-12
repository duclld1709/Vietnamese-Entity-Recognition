import pytest
import torch

from tests.conftest import V1_CHECKPOINT, tiny_encoder
from vietner.encoder import mean_pool
from vietner.labels import LEGACY_TAGS, VLSP_TAGS
from vietner.models import NERTagger, TaggerConfig


def v1_group_embeddings(tokens, embeddings):
    """Verbatim logic of v1's src/preprocessing.group_embeddings."""

    word_embeddings, current = [], []
    for token, emb in zip(tokens, embeddings, strict=True):
        if token in ["<s>", "</s>"]:
            continue
        current.append(emb)
        if not token.endswith("@@"):
            word_embeddings.append(torch.mean(torch.stack(current), dim=0))
            current = []
    if current:
        word_embeddings.append(torch.mean(torch.stack(current), dim=0))
    return word_embeddings


def test_word_embeddings_match_v1_grouping(encoder):
    words = ["Nguyễn_Văn_Nam", "làm_việc", "ở", "Hà_Nội", "."]
    tok = encoder.tokenizer
    ids = [tok.cls_token_id, *tok.convert_tokens_to_ids(tok.tokenize(" ".join(words))), tok.sep_token_id]
    with torch.no_grad():
        hidden = encoder.transformer(input_ids=torch.tensor([ids])).last_hidden_state[0]
        expected = torch.stack(v1_group_embeddings(tok.convert_ids_to_tokens(ids), hidden))
        actual = encoder.encode([words])[0]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_mean_pool():
    hidden = torch.arange(12, dtype=torch.float).view(1, 4, 3)
    index = torch.tensor([[-1, 0, 0, 1]])
    pooled, mask = mean_pool(hidden, index, 3)
    torch.testing.assert_close(pooled[0, 0], (hidden[0, 1] + hidden[0, 2]) / 2)
    torch.testing.assert_close(pooled[0, 1], hidden[0, 3])
    assert mask.tolist() == [[True, True, False]]


def test_long_inputs_are_chunked_not_crashing():
    enc = tiny_encoder(max_subwords=10)
    words = [f"từ{i:03d}" for i in range(30)]  # 2 subwords each -> 60 subwords
    chunks = enc.chunk(words)
    assert all(sum(len(enc.subword_ids(w)) for w in c) <= 10 for c in chunks)
    assert [w for c in chunks for w in c] == words
    with pytest.raises(ValueError):
        enc.batch_inputs([words])
    out = enc.encode([words, ["ngắn"]], batch_size=2)
    assert out[0].shape == (30, 32) and out[1].shape == (1, 32)
    # each chunk is encoded independently
    torch.testing.assert_close(out[0][:5], enc.encode([chunks[0]])[0])


def test_encode_handles_empty_input(encoder):
    out = encoder.encode([[], ["xin_chào"]])
    assert out[0].shape == (0, 32) and out[1].shape == (1, 32)


def test_loads_real_v1_checkpoint(encoder768):
    tagger, meta = NERTagger.load(V1_CHECKPOINT, encoder=encoder768)
    assert meta == {"source": "v1-state-dict"}
    assert tagger.tags == LEGACY_TAGS and tagger.config.mode == "probe"
    out = tagger.predict([["Nguyễn_Văn_Nam", "ở", "Hà_Nội"], [], ["a"] * 300])
    assert [len(x) for x in out] == [3, 0, 300]
    assert all(t in LEGACY_TAGS for seq in out for t in seq)


def test_probe_checkpoint_stores_only_the_head(tmp_path, encoder):
    model = NERTagger(encoder, TaggerConfig(tags=VLSP_TAGS, mode="probe"))
    model.save(tmp_path / "probe.pt", epoch=2)
    payload = torch.load(tmp_path / "probe.pt", weights_only=True)
    assert payload["encoder"] is None
    loaded, meta = NERTagger.load(tmp_path / "probe.pt", encoder=encoder)
    assert meta == {"epoch": 2} and loaded.tags == VLSP_TAGS
    torch.testing.assert_close(loaded.head.embed2tag.weight, model.head.embed2tag.weight)
    assert not any(p.requires_grad for p in loaded.encoder.parameters())


def test_finetune_checkpoint_roundtrip(tmp_path):
    enc = tiny_encoder(seed=1)
    model = NERTagger(enc, TaggerConfig(mode="finetune"))
    with torch.no_grad():
        next(enc.transformer.parameters()).add_(1.0)  # pretend PhoBERT was fine-tuned
    model.save(tmp_path / "ft.pt")
    loaded, _ = NERTagger.load(tmp_path / "ft.pt", encoder=tiny_encoder(seed=1))
    words = [["Hà_Nội", "đẹp"]]
    torch.testing.assert_close(loaded.encoder.encode(words)[0], model.encoder.encode(words)[0])


def test_finetune_loss_backpropagates_into_encoder():
    model = NERTagger(tiny_encoder(), TaggerConfig(mode="finetune"))
    loss = model.loss([["Hà_Nội", "đẹp"], ["Nam"]], [[5, 0], [1]])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.transformer.embeddings.word_embeddings.weight.grad is not None

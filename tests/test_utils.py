from gpt2_classifier.utils import generate_response


def test_generate_response_does_not_crash_on_pos_emb_access(tiny_gpt_model_real_vocab):
    # Regression test: previously used model.pos_emb.shape[0], but pos_emb is
    # an nn.Embedding (no .shape), causing an AttributeError on every call.
    # Uses the real GPT-2 vocab size since generate_response tokenizes with
    # the real tiktoken "gpt2" encoding.
    response = generate_response("hello", tiny_gpt_model_real_vocab, max_new_tokens=2)
    assert isinstance(response, str)
    assert len(response) > 0

    
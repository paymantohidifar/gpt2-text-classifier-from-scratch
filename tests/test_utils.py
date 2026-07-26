from gpt2_classifier.utils import generate_response, plot_results, plot_values


def test_generate_response_does_not_crash_on_pos_emb_access(tiny_gpt_model_real_vocab):
    # Regression test: previously used model.pos_emb.shape[0], but pos_emb is
    # an nn.Embedding (no .shape), causing an AttributeError on every call.
    # Uses the real GPT-2 vocab size since generate_response tokenizes with
    # the real tiktoken "gpt2" encoding.
    response = generate_response("hello", tiny_gpt_model_real_vocab, max_new_tokens=2)
    assert isinstance(response, str)
    assert len(response) > 0


def test_plot_values_writes_to_given_output_dir(tmp_path):
    output_path = plot_values(
        epochs_seen=[0, 1, 2],
        examples_seen=[0, 10, 20],
        train_values=[1.0, 0.8, 0.6],
        val_values=[1.1, 0.9, 0.7],
        label="loss",
        output_dir=tmp_path,
    )
    assert output_path.parent == tmp_path
    assert output_path.exists()


def test_plot_results_writes_all_metric_plots(tmp_path):
    plot_results(
        num_epochs=1,
        train_losses=[1.0, 0.8],
        val_losses=[1.1, 0.9],
        train_accs=[0.5],
        val_accs=[0.6],
        train_precisions=[0.4],
        val_precisions=[0.5],
        train_roc_aucs=[0.7],
        val_roc_aucs=[0.6],
        train_pr_aucs=[0.3],
        val_pr_aucs=[0.4],
        examples_seen=100,
        output_dir=tmp_path,
    )

    expected_files = {
        "loss-plot.pdf",
        "accuracy-plot.pdf",
        "precision-plot.pdf",
        "roc_auc-plot.pdf",
        "pr_auc-plot.pdf",
    }
    assert expected_files.issubset({p.name for p in tmp_path.iterdir()})

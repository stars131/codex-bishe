"""
Model unit tests.
"""
import pytest
import torch

from src.models.fusion_net import create_model


class TestFusionNetEncoders:
    @pytest.mark.parametrize("encoder_type", ["mlp", "cnn", "lstm", "transformer"])
    def test_encoder_forward(self, sample_data, sample_dims, encoder_type):
        model = create_model(
            model_type="fusion_net",
            traffic_dim=sample_dims["source1_dim"],
            log_dim=sample_dims["source2_dim"],
            num_classes=sample_dims["num_classes"],
            config={
                "hidden_dim": sample_dims["hidden_dim"],
                "dropout": 0.1,
                "encoder_type": encoder_type,
                "fusion_type": "attention",
                "num_layers": 2,
                "num_heads": 4,
            },
        )
        model.eval()

        with torch.no_grad():
            logits, attention = model(sample_data["source1"], sample_data["source2"])

        assert logits.shape == (sample_dims["batch_size"], sample_dims["num_classes"])
        assert attention.shape[0] == sample_dims["batch_size"]


class TestFusionMethods:
    @pytest.mark.parametrize("fusion_type", ["attention", "multi_head", "cross", "gated", "bilinear", "concat"])
    def test_fusion_forward(self, sample_data, sample_dims, fusion_type):
        model = create_model(
            model_type="fusion_net",
            traffic_dim=sample_dims["source1_dim"],
            log_dim=sample_dims["source2_dim"],
            num_classes=sample_dims["num_classes"],
            config={
                "hidden_dim": sample_dims["hidden_dim"],
                "dropout": 0.1,
                "encoder_type": "mlp",
                "fusion_type": fusion_type,
                "num_layers": 2,
                "num_heads": 4,
            },
        )
        model.eval()

        with torch.no_grad():
            logits, _ = model(sample_data["source1"], sample_data["source2"])

        assert logits.shape == (sample_dims["batch_size"], sample_dims["num_classes"])
        assert not torch.isnan(logits).any()

    def test_three_source_attention_forward(self, sample_dims):
        batch_size = sample_dims["batch_size"]
        source_dims = [sample_dims["source1_dim"], sample_dims["source2_dim"], 7]

        model = create_model(
            model_type="fusion_net",
            traffic_dim=source_dims[0],
            log_dim=source_dims[1],
            num_classes=sample_dims["num_classes"],
            config={
                "hidden_dim": sample_dims["hidden_dim"],
                "dropout": 0.1,
                "encoder_type": "mlp",
                "fusion_type": "attention",
                "num_layers": 2,
                "num_heads": 4,
                "source_dims": source_dims,
            },
        )
        model.eval()

        with torch.no_grad():
            logits, attention = model(
                torch.randn(batch_size, source_dims[0]),
                torch.randn(batch_size, source_dims[1]),
                torch.randn(batch_size, source_dims[2]),
            )

        assert logits.shape == (batch_size, sample_dims["num_classes"])
        assert attention.shape == (batch_size, 3)

    def test_three_source_cross_raises(self, sample_dims):
        source_dims = [sample_dims["source1_dim"], sample_dims["source2_dim"], 7]
        model = create_model(
            model_type="fusion_net",
            traffic_dim=source_dims[0],
            log_dim=source_dims[1],
            num_classes=sample_dims["num_classes"],
            config={
                "hidden_dim": sample_dims["hidden_dim"],
                "encoder_type": "mlp",
                "fusion_type": "cross",
                "source_dims": source_dims,
            },
        )

        with pytest.raises(ValueError, match="only supports exactly two sources"):
            model(
                torch.randn(sample_dims["batch_size"], source_dims[0]),
                torch.randn(sample_dims["batch_size"], source_dims[1]),
                torch.randn(sample_dims["batch_size"], source_dims[2]),
            )


class TestSingleSourceNet:
    def test_forward(self, sample_data, sample_dims):
        model = create_model(
            model_type="single_source",
            traffic_dim=sample_dims["source1_dim"],
            log_dim=sample_dims["source2_dim"],
            num_classes=sample_dims["num_classes"],
            config={
                "hidden_dim": sample_dims["hidden_dim"],
                "dropout": 0.1,
                "encoder_type": "mlp",
                "num_layers": 2,
            },
        )
        model.eval()

        combined = torch.cat([sample_data["source1"], sample_data["source2"]], dim=1)
        with torch.no_grad():
            logits = model(combined)

        assert logits.shape == (sample_dims["batch_size"], sample_dims["num_classes"])


class TestEnsembleFusionNet:
    def test_forward(self, sample_data, sample_dims):
        model = create_model(
            model_type="ensemble",
            traffic_dim=sample_dims["source1_dim"],
            log_dim=sample_dims["source2_dim"],
            num_classes=sample_dims["num_classes"],
            config={
                "hidden_dim": sample_dims["hidden_dim"],
                "dropout": 0.1,
                "fusion_types": ["attention", "gated"],
            },
        )
        model.eval()

        with torch.no_grad():
            logits, attention = model(sample_data["source1"], sample_data["source2"])

        assert logits.shape == (sample_dims["batch_size"], sample_dims["num_classes"])
        assert attention.shape[0] == sample_dims["batch_size"]


class TestModelFactory:
    def test_invalid_model_type(self, sample_dims):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(
                model_type="invalid_type",
                traffic_dim=sample_dims["source1_dim"],
                log_dim=sample_dims["source2_dim"],
                num_classes=sample_dims["num_classes"],
            )

    def test_gradient_flow(self, sample_data, sample_dims):
        model = create_model(
            model_type="fusion_net",
            traffic_dim=sample_dims["source1_dim"],
            log_dim=sample_dims["source2_dim"],
            num_classes=sample_dims["num_classes"],
            config={"hidden_dim": sample_dims["hidden_dim"], "encoder_type": "mlp", "fusion_type": "attention"},
        )
        model.train()

        logits, _ = model(sample_data["source1"], sample_data["source2"])
        loss = torch.nn.CrossEntropyLoss()(logits, sample_data["labels"])
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

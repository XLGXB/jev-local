import torch
import pytest
from decision_ft.objectives import (soft_ce, proper_score, rlcd_loss,
    ranked_error, reference_kl, check)


def fixture():
    z = torch.tensor([[0.2, -0.1, 0.8, 0.0], [0.4, -0.3, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    target = torch.tensor([[0., 0., 1., 0.], [1., 0., 0., 0.]])
    kind = torch.tensor([1, 2])
    return z, target, mask, kind


def test_soft_ce_matches_hard_ce():
    z, target, mask, _ = fixture()
    actual = soft_ce(z, target, mask)
    expected = torch.stack([torch.nn.functional.cross_entropy(z[0, :3][None], torch.tensor([2])),
                            torch.nn.functional.cross_entropy(z[1, :2][None], torch.tensor([0]))])
    assert torch.allclose(actual, expected)


def test_all_supervised_logits_receive_finite_gradients():
    z, target, mask, _ = fixture()
    soft_ce(z, target, mask).mean().backward()
    assert torch.isfinite(z.grad).all() and z.grad[mask].abs().sum() > 0
    assert torch.equal(z.grad[~mask], torch.zeros_like(z.grad[~mask]))


def test_rl_gradient_is_nonzero_and_padding_is_zero():
    torch.manual_seed(13)
    z, target, mask, kind = fixture()
    loss, stats = rlcd_loss(z, target, mask, kind, group_size=32, reference_logits=z.detach().clone())
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    assert z.grad[mask].abs().sum() > 0 and torch.isfinite(z.grad).all()
    assert torch.equal(z.grad[~mask], torch.zeros_like(z.grad[~mask]))


def test_rps_respects_ordinal_distance():
    mask = torch.ones((1, 3), dtype=torch.bool)
    target = torch.tensor([[1., 0., 0.]])
    near = ranked_error(torch.tensor([[0., 1., 0.]]), target, mask)
    far = ranked_error(torch.tensor([[0., 0., 1.]]), target, mask)
    assert far > near


def test_proper_score_prefers_correct_report_on_soft_target():
    mask = torch.ones((1, 3), dtype=torch.bool)
    target = torch.tensor([[0.6, 0.3, 0.1]])
    ideal = proper_score(target.log(), target, mask, torch.tensor([0]))
    wrong = proper_score(torch.tensor([[0.1, 0.3, 0.6]]).log(), target, mask, torch.tensor([0]))
    assert ideal > wrong


def test_reference_kl_zero_for_identical_logits():
    z, _, mask, _ = fixture()
    assert torch.allclose(reference_kl(z, z, mask), torch.zeros(2), atol=1e-6)


def test_rl_requires_reference_when_kl_enabled():
    z, t, m, k = fixture()
    with pytest.raises(ValueError):
        rlcd_loss(z, t, m, k)


def test_uniform_reports_have_finite_reward():
    _, t, m, k = fixture()
    z = torch.zeros_like(t, requires_grad=True)
    loss, _ = rlcd_loss(z, t, m, k, reference_logits=z.detach())
    assert torch.isfinite(loss)


def test_score_function_mc_matches_reparameterized_gradient():
    # Independent numerical verification of the policy-gradient implementation.
    z = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True)
    t = torch.tensor([[0.1, 0.2, 0.7]])
    mask = torch.ones_like(z, dtype=torch.bool)
    k = torch.tensor([0])
    torch.manual_seed(92)
    loss, _ = rlcd_loss(z, t, mask, k, group_size=32768, sigma=0.25,
                       rl_weight=1., ce_weight=0., kl_weight=0.)
    pg = torch.autograd.grad(loss, z)[0]
    torch.manual_seed(92)
    noise = torch.randn((32768,) + z.shape)
    pathwise = -proper_score(z.unsqueeze(0) + 0.25 * noise, t, mask, k).mean()
    exact_mc = torch.autograd.grad(pathwise, z)[0]
    assert torch.allclose(pg, exact_mc, atol=0.015, rtol=0.10), (pg, exact_mc)

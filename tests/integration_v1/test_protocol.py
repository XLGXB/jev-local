import asyncio
from copy import deepcopy
import unittest
from types import SimpleNamespace

from decision_deploy.protocol import validate_envelope, validate_reply
from jevlocal.integration.protocol import legacy_to_envelope, reply_to_legacy


def envelope():
    return {"request_id": "r", "observation_id": "o", "state": "当前可见状态",
            "questions": {"boolean": {"type": "noul", "instructions": "是否可见？"},
                          "pick": {"type": "choice", "instructions": "选择", "criteria": {"b": "乙", "a": "甲"}},
                          "quality": {"type": "score", "instructions": "评分", "criteria": ["低", "高"]}}}


def reply(request=None):
    r = request or envelope()
    return {"request_id": r["request_id"], "observation_id": r["observation_id"],
            "runtime_fingerprint": "test-only-fingerprint", "answers": {
                "boolean": {"selected_key": "true", "probabilities": {"false": .2, "true": .8}, "p_selected": .8, "calibrated": True},
                "pick": {"selected_key": "a", "probabilities": {"a": .9, "b": .1}, "p_selected": .9, "calibrated": True},
                "quality": {"selected_key": "1", "probabilities": {"0": .25, "1": .75}, "p_selected": .75, "calibrated": False}}}


class ProtocolTests(unittest.TestCase):
    def test_valid_three_types(self):
        self.assertEqual(validate_envelope(envelope())["boolean"], ["false", "true"])
        validate_reply(envelope(), reply())
        value = reply(); value["inference_total_ms"] = 2.5
        value["answers"]["boolean"]["temperature"] = 1.2
        validate_reply(envelope(), value)

    def test_legacy_semantics_are_keyed_and_shadow_only(self):
        result = reply_to_legacy(envelope(), reply())
        self.assertEqual(result["results"]["boolean"]["probabilities"], [.8, .2])
        self.assertEqual(result["results"]["boolean"]["noul"], .8)
        self.assertEqual(result["results"]["pick"]["probabilities"], [.1, .9])
        self.assertEqual(result["results"]["pick"]["choice"], 1)
        self.assertEqual(result["results"]["quality"]["score"], .75)
        self.assertTrue(result["shadow"])
        self.assertFalse(result["execution_authorized"])

    def test_no_mutation_or_implicit_ids(self):
        r = envelope()
        legacy = {k: deepcopy(r[k]) for k in ("state", "questions")}
        before = deepcopy(legacy)
        converted = legacy_to_envelope(legacy, request_id="r", observation_id="o")
        self.assertEqual(converted, r)
        converted["questions"]["pick"]["criteria"]["b"] = "changed"
        self.assertEqual(legacy, before)

    def test_extra_envelope_metadata_rejected(self):
        for key in ["schema_version", "policy_hash", "_meta"]:
            r = envelope(); r[key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError): validate_envelope(r)

    def test_bad_requests_rejected(self):
        cases = []
        r = envelope(); r["questions"] = {}; cases.append(r)
        r = envelope(); r["request_id"] = " "; cases.append(r)
        r = envelope(); r["state"] = float("nan"); cases.append(r)
        r = envelope(); r["questions"]["boolean"]["label"] = True; cases.append(r)
        r = envelope(); r["questions"]["pick"]["criteria"] = {"a": "only"}; cases.append(r)
        r = envelope(); r["questions"]["pick"]["criteria"] = {str(i): str(i) for i in range(17)}; cases.append(r)
        r = envelope(); r["questions"] = {str(i): r["questions"]["boolean"] for i in range(9)}; cases.append(r)
        r = envelope(); r["state"] = "x" * (128 * 1024); cases.append(r)
        r = envelope(); r["questions"]["pick"]["criteria"]["a"] = {}; cases.append(r)
        for i, request in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError): validate_envelope(request)

    def test_bad_responses_rejected(self):
        mutations = [
            lambda r: r.update(private_text="unexpected internal context"),
            lambda r: r.update(inference_total_ms="not a timing"),
            lambda r: r["answers"]["boolean"].update(unexpected="private"),
            lambda r: r["answers"]["boolean"].update(temperature=0),
            lambda r: r.update(observation_id="stale"),
            lambda r: r.update(request_id="other"),
            lambda r: r["answers"].pop("pick"),
            lambda r: r["answers"].update(extra={}),
            lambda r: r["answers"]["boolean"]["probabilities"].pop("false"),
            lambda r: r["answers"]["boolean"].update(selected_key="false"),
            lambda r: r["answers"]["boolean"].update(selected_key=[]),
            lambda r: r["answers"]["boolean"].update(p_selected=.3),
            lambda r: r["answers"]["boolean"].update(calibrated=1),
            lambda r: r["answers"]["boolean"].update(probabilities={"true": float("nan"), "false": .2}),
            lambda r: r["answers"]["boolean"].update(probabilities={"true": True, "false": False}),
            lambda r: r["answers"]["boolean"].update(probabilities={"true": .8, "false": .3}),
        ]
        for i, mutate in enumerate(mutations):
            value = reply(); mutate(value)
            with self.subTest(case=i), self.assertRaises(ValueError): validate_reply(envelope(), value)


class HTTPBoundaryTests(unittest.TestCase):
    def test_asgi_contract_without_network_or_model(self):
        import httpx
        from decision_deploy.server import create_app
        class Predictor:
            runtime = SimpleNamespace(fingerprint="test-only-fingerprint")
            calls = 0
            invalid = False
            def predict(self, request):
                self.calls += 1
                return {} if self.invalid else reply()
        async def scenario():
            predictor = Predictor()
            app = create_app(predictor, "test-only-token-" + "x" * 32)
            headers = {"Authorization": "Bearer test-only-token-" + "x" * 32}
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
                self.assertEqual((await client.post("/v1/decide", json=envelope())).status_code, 401)
                self.assertEqual((await client.post("/v1/decide", json=envelope(), headers={**headers, "Origin": "http://example.test"})).status_code, 403)
                bad = envelope(); bad["questions"] = {}
                self.assertEqual((await client.post("/v1/decide", json=bad, headers=headers)).status_code, 400)
                self.assertEqual(predictor.calls, 0)
                response = await client.post("/v1/decide", json=envelope(), headers=headers)
                self.assertEqual(response.status_code, 200)
                validate_reply(envelope(), response.json())
                predictor.invalid = True
                self.assertEqual((await client.post("/v1/decide", json=envelope(), headers=headers)).status_code, 503)
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

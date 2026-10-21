import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("mcp"), "optional MCP SDK is required")
class MCPContextTests(unittest.TestCase):
    def test_explicit_tool_surface_without_server_or_global_configuration(self):
        from jevlocal.integration.mcp_server import create_server
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)/"store"
            server = create_server(store)
            tools = asyncio.run(server.list_tools())
            self.assertEqual({tool.name for tool in tools}, {
                "rank_evidence", "filter_tool_output", "suggest_skills", "restore_evidence", "scope_evidence"})
            self.assertTrue(all(tool.annotations.openWorldHint is False for tool in tools))
            self.assertFalse(store.exists())
            by_name = {tool.name: tool for tool in tools}
            self.assertIn("candidates", by_name["rank_evidence"].inputSchema["properties"])
            self.assertNotIn("store_dir", by_name["rank_evidence"].inputSchema["properties"])
            self.assertNotIn("score_callback", by_name["rank_evidence"].inputSchema["properties"])
            self.assertTrue(by_name["restore_evidence"].annotations.readOnlyHint)
            self.assertTrue(by_name['scope_evidence'].annotations.readOnlyHint)
            self.assertIn('target', by_name['scope_evidence'].inputSchema['properties'])
            self.assertIn('bindings', by_name['scope_evidence'].inputSchema['properties'])
            self.assertNotIn('store_dir', by_name['scope_evidence'].inputSchema['properties'])
            self.assertNotIn('known_secrets', by_name['scope_evidence'].inputSchema['properties'])

    def test_configured_store_is_not_resolved_before_symlink_validation(self):
        from jevlocal.integration.mcp_server import create_server
        with tempfile.TemporaryDirectory() as directory:
            with patch("pathlib.Path.is_symlink", return_value=True):
                with self.assertRaisesRegex(ValueError, "non-symlink"):
                    create_server(Path(directory)/"store")
            with self.assertRaisesRegex(ValueError, "absolute"):
                create_server("relative-store")

    def test_scope_tool_invokes_pure_preparation_without_store_or_rank(self):
        from jevlocal.integration.mcp_server import create_server
        from jevlocal.integration.evidence_scope import candidate_digest
        with tempfile.TemporaryDirectory() as directory:
            store=Path(directory)/'not-created'
            server=create_server(store)
            candidate={'id':'a','source':'owned-note','text':'Existing result'}
            arguments={'query':'Existing result','candidates':[candidate],
                'target':{'object_id':'object-A','revision':'r2'},
                'bindings':{'a':{'object_id':'object-A','revision':None,
                                  'candidate_sha256':candidate_digest(candidate)}}}
            with patch('jevlocal.integration.context_tools._store',side_effect=AssertionError('no store')), \
                 patch('jevlocal.integration.context_tools.rank_evidence',side_effect=AssertionError('no ranking')):
                content=asyncio.run(server.call_tool('scope_evidence',arguments))
            result=json.loads(content[0].text)
            self.assertEqual(result['counts'],{'total':1,'match':0,'mismatch':0,'unknown':1})
            self.assertEqual(result['status'],'no_matches')
            self.assertIsNone(result['rank_input'])
            self.assertFalse(result['origin_authenticated'])
            self.assertFalse(store.exists())


if __name__ == "__main__":
    unittest.main()

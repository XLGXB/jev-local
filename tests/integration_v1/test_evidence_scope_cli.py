from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jevlocal.integration.cli import main
from jevlocal.integration.context_tools import rank_evidence


def arguments():
    candidates=[{'id':key,'source':key+'.txt','start_line':7,
                 'text':'task result\r\npassword=fake-private-value\r\n'}
                for key in ('matched','other','old','unknown')]
    pairs=[('obj-A','rev-2'),('obj-B','rev-2'),('obj-A','rev-1'),('obj-A',None)]
    bindings={}
    for c,(obj,rev) in zip(candidates,pairs):
        raw=json.dumps(c,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        bindings[c['id']]={'object_id':obj,'revision':rev,'candidate_sha256':hashlib.sha256(raw).hexdigest()}
    return dict(query='task result',candidates=candidates,target={'object_id':'obj-A','revision':'rev-2'},
                bindings=bindings,max_chars=30000,rank_max_chars=12000)


class ScopeCliTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'original.json';self.scope=self.root/'scope.json';self.rank=self.root/'rank.json'
        self.args=arguments();self.write_source()

    def tearDown(self):self.temp.cleanup()

    def write_source(self):self.source.write_text(json.dumps(self.args),encoding='utf-8')

    def invoke(self,command):
        output=io.StringIO()
        with redirect_stdout(output):code=main(command)
        return code,json.loads(output.getvalue())

    def prepare(self,extra=()):
        return self.invoke(['scope-evidence','--input',str(self.source),'--output',str(self.scope),
                            '--rank-input-output',str(self.rank),*extra])

    def verify(self,extra=()):
        return self.invoke(['verify-scoped-evidence','--input',str(self.source),'--prepared',str(self.scope),
                            '--rank-input',str(self.rank),*extra])

    def test_prepare_real_lexical_composition_and_verify(self):
        original=self.source.read_bytes()
        code,prepared=self.prepare();self.assertEqual(code,0)
        self.assertEqual(prepared['counts'],{'total':4,'match':1,'mismatch':2,'unknown':1})
        derived=json.loads(self.rank.read_bytes());self.assertEqual(derived,prepared['rank_input'])
        self.assertEqual([c['id'] for c in derived['candidates']],['matched'])
        self.assertEqual(derived['candidates'][0]['start_line'],7)
        self.assertNotIn('fake-private-value',self.scope.read_text())
        code,primary=self.invoke(['context','rank_evidence','--input',str(self.rank)])
        self.assertEqual(code,0);self.assertEqual(primary,rank_evidence(**derived))
        self.assertFalse(primary['execution_authorized'])
        code,report=self.verify();self.assertEqual(code,0);self.assertEqual(report['scope']['counts'],prepared['counts'])
        self.assertFalse(report['execution_authorized']);self.assertEqual(self.source.read_bytes(),original)

    def test_no_matches_produces_no_derived_file_or_inference(self):
        self.args['target']['object_id']='unmatched';self.write_source()
        with patch('jevlocal.integration.context_shadow.start_shadow',side_effect=AssertionError('must not infer')):
            code,result=self.prepare()
        self.assertEqual(code,0);self.assertEqual(result['status'],'no_matches');self.assertFalse(self.rank.exists())
        code,report=self.invoke(['verify-scoped-evidence','--input',str(self.source),'--prepared',str(self.scope)])
        self.assertEqual(code,0);self.assertIsNone(report['scope']['rank_input_sha256'])

    def test_output_collision_checked_before_any_publication(self):
        self.rank.write_text('user artifact',encoding='utf-8')
        code,result=self.prepare();self.assertEqual(code,2);self.assertFalse(self.scope.exists())
        self.assertEqual(self.rank.read_text(),'user artifact')
        before=self.source.read_bytes()
        code,_=self.invoke(['scope-evidence','--input',str(self.source),'--output',str(self.source)])
        self.assertEqual(code,2);self.assertEqual(self.source.read_bytes(),before)

    def test_identical_two_output_paths_rejected(self):
        code,_=self.invoke(['scope-evidence','--input',str(self.source),'--output',str(self.scope),
                            '--rank-input-output',str(self.scope)])
        self.assertEqual(code,2);self.assertFalse(self.scope.exists())

    def test_budget_failure_publishes_neither_file(self):
        self.args['max_chars']=35;self.write_source()
        code,result=self.prepare();self.assertEqual(code,2);self.assertEqual(result['status'],'requires_larger_budget')
        self.assertFalse(self.scope.exists());self.assertFalse(self.rank.exists())

    def test_tampered_scope_or_derived_input_rejected(self):
        self.prepare();saved=self.scope.read_bytes()
        v=json.loads(saved);v['counts']['total']=1;self.scope.write_text(json.dumps(v))
        self.assertEqual(self.verify()[0],2)
        self.scope.write_bytes(saved);v=json.loads(self.rank.read_bytes());v['candidates'][0]['text']='changed'
        self.rank.write_text(json.dumps(v));self.assertEqual(self.verify()[0],2)

    def test_combined_replay_needs_both_primary_and_shadow(self):
        self.prepare()
        code,result=self.verify(['--primary',str(self.root/'missing.json')]);self.assertEqual(code,2)
        self.assertEqual(result['status'],'rejected')

    def test_malformed_or_oversized_json_has_no_partial_output(self):
        for payload in ['{"query":"a","query":"b"}','{"query":NaN}','['*1500+']'*1500,' '*((4*1024*1024)+1)]:
            self.source.write_text(payload,encoding='utf-8')
            code,result=self.prepare();self.assertEqual(code,2);self.assertEqual(result['status'],'rejected')
            self.assertFalse(self.scope.exists());self.assertFalse(self.rank.exists())

    def test_artifact_reader_rejects_float_overflow_before_consumer(self):
        from jevlocal.integration.cli import _read_scope_artifact
        for payload in ['{"x":1e9999}', '{"nested":[-1e9999]}']:
            self.source.write_text(payload,encoding='utf-8')
            with self.assertRaises(ValueError):_read_scope_artifact(self.source)

    def test_scope_rejects_paths_callbacks_and_native_flags(self):
        self.args['store_dir']=str(self.root/'not-created');self.write_source()
        code,_=self.prepare();self.assertEqual(code,2);self.assertFalse((self.root/'not-created').exists())
        with self.assertRaises(SystemExit) as caught:self.prepare(['--native-runtime','unrequested.json'])
        self.assertEqual(caught.exception.code,2)

    def test_second_publication_failure_is_explicit_not_success(self):
        from jevlocal.integration import cli
        real=cli._write_new_atomic
        def write(path,value):
            if Path(path)==self.rank:raise OSError('sensitive-hidden-error')
            return real(path,value)
        with patch.object(cli,'_write_new_atomic',side_effect=write):code,result=self.prepare()
        self.assertEqual(code,2);self.assertEqual(result['status'],'rejected')
        self.assertTrue(self.scope.exists());self.assertFalse(self.rank.exists())
        self.assertNotIn('sensitive-hidden-error',json.dumps(result))


if __name__=='__main__':unittest.main()

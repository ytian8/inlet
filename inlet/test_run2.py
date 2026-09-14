"""CPU tests for the Run 2 information boundary and generation instrumentation."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from inlet.aligned_protocol import strip_definitions
from inlet.run2_setup import make_config, dataclass_fields
from inlet.generation_monitor import run_monitor, generation_end

class Run2Tests(unittest.TestCase):
    def test_definition_removed_without_destroying_rules_or_question(self):
        source={'lol_1': {'user_prompt_template':'{task_def}\n\n{problem}',
                          'descriptions':['Return X for yes, Y for no.'],
                          'ds_kwargs':{'split':'train'}}}
        original=copy.deepcopy(source)
        result, changed=strip_definitions(source,True)
        self.assertEqual(changed,['lol_1'])
        self.assertEqual(source,original)
        self.assertEqual(result['lol_1']['user_prompt_template'].format(problem='Question?'),'Question?')
        self.assertEqual(result['lol_1']['descriptions'],source['lol_1']['descriptions'])
    def test_unexpected_templates_fail_instead_of_deleting_question(self):
        for tpl in ['{task_def}\n{problem}','{problem}','{task_def}\n\n{problem}\nExtra']:
            with self.assertRaises(ValueError):
                strip_definitions({'lol_1':{'user_prompt_template':tpl,'descriptions':['d']}},True)
    def test_description_and_system_guards(self):
        for extra in [{'descriptions':['']},{'system_message':'{task_def}'}]:
            md={'user_prompt_template':'{task_def}\n\n{problem}','descriptions':['d'],**extra}
            with self.assertRaises(ValueError):strip_definitions({'lol_1':md},True)
    def test_run1_config_preserved_and_two_gpu_budget(self):
        source=dict(desc_slots=8,cond='cross',n_virtual_tokens=32,
                    model_dir='mistralai/Mistral-7B-Instruct-v0.2',n_train_ds=479,
                    n_points_per_task=1,n_tasks_per_batch=8,use_per_task_emb=True,
                    sft_mode='completion',global_tasks_per_step=64,lr=2.5e-5,
                    warmup_frac=.2,seed=42,train_ds_names=['lol_'+str(i) for i in range(479)],
                    eval_ds_info={'lol_999':{}},max_steps=16000)
        old=copy.deepcopy(source); cfg=make_config(source,'run2')
        self.assertEqual(source,old)
        self.assertEqual(2*cfg['n_tasks_per_batch']*cfg['grad_accum_steps'],64)
        for key in ['lr','warmup_frac','seed','train_ds_names','desc_slots','cond']:
            self.assertEqual(cfg[key],source[key])
        self.assertTrue(cfg['strip_taskdef_in_training'] and cfg['strip_taskdef_in_validation'])
        self.assertEqual(cfg['generative_val_tasks'],'')
        with self.assertRaises(ValueError):make_config(dict(source,global_tasks_per_step=128),'bad')
    def test_end_reason(self):
        self.assertEqual(generation_end([8,2],2,2),'eos')
        self.assertEqual(generation_end([8,9],2,2),'length_limit')
        self.assertEqual(generation_end([8],None,2),'other')
    def test_monitor_does_not_force_length_or_advance_rng(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.emb=torch.nn.Embedding(20,4)
                self.generation_config=SimpleNamespace(eos_token_id=2)
                self.calls=[]
            def get_input_embeddings(self):return self.emb
            def generate(self,**kw):
                self.calls.append(kw); torch.rand(1)
                return torch.tensor([[9,2]])
        class Head(torch.nn.Module):
            def forward(self,x):return torch.zeros((1,3,4))
        tok=SimpleNamespace(decode=lambda ids,**kw:' '.join(str(int(i)) for i in ids))
        model,head=Model(),Head()
        examples=[(2,0,0,{'task_embs':torch.ones((1,2,4))},torch.tensor([4,5,6]),torch.tensor([9,2]))]
        before=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as d:
            rows=run_monitor(model,head,tok,examples,d,8,16)
            self.assertEqual(len(Path(d,'generation_monitor.jsonl').read_text().splitlines()),2)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(head.training)
        self.assertEqual([c['inputs_embeds'].shape[1] for c in model.calls],[3,6])
        for c in model.calls:
            self.assertEqual(c['min_new_tokens'],0);self.assertEqual(c['min_length'],0)
        self.assertTrue(all(r['end']=='eos' and r['generated_tokens']==2 for r in rows))

if __name__=='__main__':unittest.main()

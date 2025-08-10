import os, json, random, pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
N_SCENARIOS=50
COUNTS={'RB':8,'WR':8,'TE':4,'QB':4}
PROMPT_VARIANTS=['tom','baseline']
MODELS=['gpt-5','claude-3.5-sonnet','gemini-2.5-pro']
OUTDIR=Path('runs')
OUTDIR.mkdir(exist_ok=True);(OUTDIR/'prompts').mkdir(parents=True,exist_ok=True);(OUTDIR/'outputs').mkdir(exist_ok=True);(OUTDIR/'analysis').mkdir(exist_ok=True)

players=pd.read_csv('top_240_fantasy_players.csv')
players['projected_points']=pd.to_numeric(players['projected_points'],errors='coerce').fillna(0.0)
players['name']=players['name'].astype(str);players['position']=players['position'].astype(str)
by_name=players.set_index('name').to_dict(orient='index')

def sample_roster(pool):
    r=[]
    for pos,k in COUNTS.items():
        c=[p for p in pool if p['position']==pos]
        r+=random.sample(c,k)
        pool[:]=[p for p in pool if p not in r]
    return r

def make_scenarios():
    s=[];base=players.to_dict(orient='records')
    for sid in range(N_SCENARIOS):
        pool=base.copy();random.shuffle(pool)
        A=sample_roster(pool);B=sample_roster(pool)
        s.append({'scenario_id':sid,'teamA':[p['name'] for p in A],'teamB':[p['name'] for p in B]})
    return s

PROMPT_HEADER=("You are negotiating an NFL fantasy football trade for Team A. Strictly output a single-line JSON with keys: offer_from_A (list of names), ask_from_B (list of names), rationale (string). Names must exactly match the provided rosters. Keep the JSON minified with double quotes.")

def roster_block(name,roster):
    rows=[f"{r} | {by_name[r]['position']} | {by_name[r]['projected_points']:.1f}" for r in roster]
    return f"{name} Roster (name | pos | proj):\n"+"\n".join(rows)

def render_prompt(variant,teamA,teamB):
    A=roster_block('Team A',teamA);B=roster_block('Team B',teamB)
    
    if variant=='tom':
        g=("You are the manager of Team A. Your goal is to improve the RB position as you lack depth at that position by proposing a mutually beneficial trade with Team B that Team B is likely to accept. Team B is weak in the WR position. Consider what team B needs or might accept. You may offer any players from your own team in exchange for players on Team B's roster. Carefully analyze both teams rosters, including positional strengths and weaknesses. Your trade should 1. Improve Team A's RB depth 2. Consider Team B's perspective which includes their positional needs and likely willingness to accept the trade 3. Be fair, realistic, and aligned with each team's goals. Your task is to propose a trade and clearly state which player(s) each team would give and receive. After, write a detailed explanation on why the trade benefits both teams, what needs each team is addressing and why Team B is likely to accept the trade.")
    else: 
        g=("You are the manager of Team A in a fantasy football league. Your goal is to improve your team's running back depth by proposing a trade with team B. Review both rosters and propose a trade that accomplishes your goal. Provide a detailed explanation for your trade.")
    
    c=("You may trade any number of players. Trade must not include names outside rosters. Return JSON only, no prose.")
    return f"{PROMPT_HEADER}\n\n{A}\n\n{B}\n\nTask: {g} {c}"

def write_prompts(scenarios):
    man=[]
    for s in scenarios:
        for v in PROMPT_VARIANTS:
            for m in MODELS:
                f=OUTDIR/'prompts'/f"sc{s['scenario_id']}_{v}_{m}.txt"
                f.write_text(render_prompt(v,s['teamA'],s['teamB']))
                man.append({'scenario_id':s['scenario_id'],'variant':v,'model':m,'prompt_path':str(f)})
    (OUTDIR/'prompts_manifest.json').write_text(json.dumps(man,indent=2))
    (OUTDIR/'scenarios.json').write_text(json.dumps(scenarios,indent=2))
    return man

MODEL_MAP={'gpt-5':'gpt-5','claude-3.5-sonnet':'claude-3-5-sonnet-20240620','gemini-2.5-pro':'models/gemini-2.5-pro'}

def call_openai(prompt,model_id):
    try:
        from openai import OpenAI
        r=OpenAI().chat.completions.create(model=model_id,messages=[{"role":"user","content":prompt}],response_format={"type":"json_object"},temperature=0)
        return r.choices[0].message.content
    except Exception:
        return None

def call_anthropic(prompt,model_id):
    try:
        import anthropic
        c=anthropic.Anthropic()
        m=c.messages.create(model=model_id,max_tokens=512,temperature=0,messages=[{"role":"user","content":prompt}])
        t=[]
        for p in m.content:
            if getattr(p,'type','')=='text': t.append(p.text)
        return "\n".join(t)
    except Exception:
        return None

def call_gemini(prompt,model_id):
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        g=genai.GenerativeModel(model_id)
        r=g.generate_content(prompt,generation_config={"response_mime_type":"application/json"})
        return getattr(r,'text',None) or (r.candidates[0].content.parts[0].text if r.candidates else None)
    except Exception:
        return None

def call_model(model,prompt):
    if model=='gpt-5': return call_openai(prompt,MODEL_MAP[model])
    if model=='claude-3.5-sonnet': return call_anthropic(prompt,MODEL_MAP[model])
    if model=='gemini-2.5-pro': return call_gemini(prompt,MODEL_MAP[model])
    return None

def team_points(names):
    return sum(by_name[n]['projected_points'] for n in names)

def apply_trade(teamA,teamB,trade):
    A=set(teamA);B=set(teamB);offer=set(trade.get('offer_from_A',[]));ask=set(trade.get('ask_from_B',[]))
    if not offer.issubset(A) or not ask.issubset(B): return None
    return list((A-offer)|ask),list((B-ask)|offer)

def pos_vector(names):
    order = ['QB','RB','WR','TE']
    cnt = {k:0 for k in order}
    for n in names:
        p = by_name.get(n,{}).get('position')
        if p in cnt: cnt[p] += 1
    v = [cnt[k] for k in order]
    s = sum(v) or 1
    return [x/s for x in v]

def cosine(a,b):
    import math
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0: return 0.0
    return dot/(na*nb)

IDEAL_VEC = pos_vector(sum([[k]*v for k,v in COUNTS.items()],[]))

KEYWORDS = {'need','needs','believe','likely','accept','balance','fit','perspective','goal','willing','weak','strong'}
def tom_keyword_score(text):
    if not isinstance(text,str): return 0
    t = text.lower()
    return sum(1 for w in KEYWORDS if w in t)

def fairness_score(offer_names, ask_names):
    a = sum(by_name[n]['projected_points'] for n in ask_names)
    g = sum(by_name[n]['projected_points'] for n in offer_names)
    ta = a - g
    tb = -ta
    denom = abs(ta) + abs(tb) or 1.0
    return 1 - abs(ta - tb)/denom, ta, tb

def collect_results(scenarios):
    rows=[]
    for s in scenarios:
        for v in PROMPT_VARIANTS:
            for m in MODELS:
                p = OUTDIR/'outputs'/f"sc{s['scenario_id']}_{v}_{m}.json"
                if not p.exists(): 
                    continue
                try:
                    tr = json.loads(p.read_text())
                except Exception:
                    continue
                BA = team_points(s['teamA']); BB = team_points(s['teamB'])
                applied = apply_trade(s['teamA'], s['teamB'], tr)
                if not applied:
                    continue
                A_new, B_new = applied
                AA = team_points(A_new); AB = team_points(B_new)

                fair, tAchg, tBchg = fairness_score(tr.get('offer_from_A',[]), tr.get('ask_from_B',[]))
                fit_before = cosine(pos_vector(s['teamA']), IDEAL_VEC)
                fit_after  = cosine(pos_vector(A_new),    IDEAL_VEC)
                kw = tom_keyword_score(tr.get('reasoning') or tr.get('rationale'))

                rows.append({
                    'scenario_id': s['scenario_id'], 'variant': v, 'model': m,
                    'delta_A': AA-BA, 'delta_B': AB-BB, 'mutual_gain': int((AA-BA)>0 and (AB-BB)>0),
                    'fairness': fair, 'teamA_change': tAchg, 'teamB_change': tBchg,
                    'fit_before': fit_before, 'fit_after': fit_after, 'fit_delta': fit_after - fit_before,
                    'tom_kw': kw
                })
    return pd.DataFrame(rows)

if __name__=='__main__':
    sc=make_scenarios();man=write_prompts(sc)
    for e in man:
        outp=OUTDIR/'outputs'/f"sc{e['scenario_id']}_{e['variant']}_{e['model']}.json"
        if outp.exists(): continue
        txt=call_model(e['model'],Path(e['prompt_path']).read_text())
        if not txt: continue
        line=txt.strip().splitlines()[0]
        try:
            obj=json.loads(line);outp.write_text(json.dumps(obj))
        except Exception:
            import re
            m=re.search(r"\{[\s\S]*\}",txt)
            if m:
                try: obj=json.loads(m.group(0));outp.write_text(json.dumps(obj))
                except Exception: pass
    df=collect_results(sc)
    if not df.empty:
        df.to_csv(OUTDIR/'analysis'/'results.csv',index=False)
        import matplotlib.pyplot as plt
        plt.figure();df.boxplot(column='delta_A',by='variant');plt.suptitle('');plt.title('Team A Improvement by Prompt Variant');plt.ylabel('Projected Points Delta');plt.savefig(OUTDIR/'analysis'/'boxplot_delta_A.png',bbox_inches='tight')
        r=df.groupby('variant')['mutual_gain'].mean().reset_index()
        plt.figure();plt.bar(r['variant'],r['mutual_gain']);plt.ylabel('Mutual Gain Rate');plt.xticks(rotation=15);plt.savefig(OUTDIR/'analysis'/'mutual_gain_rate.png',bbox_inches='tight')
    # fairness by variant
plt.figure()
df.boxplot(column='fairness', by='variant'); plt.suptitle('')
plt.title('Fairness by Prompt Variant'); plt.ylabel('Fairness')
plt.savefig(OUTDIR/'analysis'/'fairness_by_variant.png', bbox_inches='tight')

# ToM keyword proxy by variant
g = df.groupby('variant')['tom_kw'].mean().reset_index()
plt.figure()
plt.bar(g['variant'], g['tom_kw']); plt.ylabel('Mean ToM Keyword Count'); plt.xticks(rotation=15)
plt.savefig(OUTDIR/'analysis'/'tom_kw_by_variant.png', bbox_inches='tight')

# Scatter: ToM proxy vs fairness
plt.figure()
plt.scatter(df['tom_kw'], df['fairness'])
plt.xlabel('ToM Keyword Count'); plt.ylabel('Fairness')
plt.savefig(OUTDIR/'analysis'/'scatter_tomkw_vs_fairness.png', bbox_inches='tight')

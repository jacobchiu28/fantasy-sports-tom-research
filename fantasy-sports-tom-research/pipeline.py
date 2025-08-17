import os, json, random, pandas as pd
import math
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
N_SCENARIOS=50
COUNTS={'RB':8,'WR':8,'TE':4,'QB':4}
PROMPT_VARIANTS=['tom','baseline']
MODELS=['gpt-4o','claude-3.5-sonnet','gemini-2.5-pro']
OUTDIR=Path('runs')
OUTDIR.mkdir(exist_ok=True);(OUTDIR/'prompts').mkdir(parents=True,exist_ok=True);(OUTDIR/'outputs').mkdir(exist_ok=True);(OUTDIR/'analysis').mkdir(exist_ok=True)

players=pd.read_csv('top_240_fantasy_players.csv')
players['projected_points']=pd.to_numeric(players['projected_points'],errors='coerce').fillna(0.0)
players['name']=players['name'].astype(str);players['position']=players['position'].astype(str)
by_name=players.set_index('name').to_dict(orient='index')

def sample_roster(pool, max_total=14):
    r = []
    remaining_slots = max_total
    min_requirements = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
    for pos, min_count in min_requirements.items():
        candidates = [p for p in pool if p['position'] == pos and p not in r]
        if len(candidates) < min_count:
            continue
        selected = random.sample(candidates, min_count)
        r.extend(selected)
        remaining_slots -= min_count
    while remaining_slots > 0:
        available_positions = []
        for pos, max_pos in COUNTS.items():
            current_count = len([p for p in r if p['position'] == pos])
            if current_count < max_pos:
                candidates = [p for p in pool if p['position'] == pos and p not in r]
                if candidates:
                    available_positions.extend(candidates)
        
        if not available_positions:
            break
            
        chosen = random.choice(available_positions)
        r.append(chosen)
        remaining_slots -= 1
    
    return r

def load_or_make_scenarios(max_roster=14):
    sc_path = OUTDIR / "scenarios.json"
    if sc_path.exists():
        return json.loads(sc_path.read_text())
    else:
        s = []
        base = players.to_dict(orient='records')
        for sid in range(N_SCENARIOS):
            pool = base.copy()
            random.shuffle(pool)
            A = sample_roster(pool, max_total=max_roster)
            B = sample_roster(pool, max_total=max_roster)
            s.append({
                'scenario_id': sid,
                'teamA': [p['name'] for p in A],
                'teamB': [p['name'] for p in B]
            })
        sc_path.write_text(json.dumps(s, indent=2))
        return s

PROMPT_HEADER=("You are negotiating a fantasy football trade for Team A. Strictly output a single-line JSON with keys: offer_from_A (list of names), ask_from_B (list of names), reasoning (string). Names must exactly match the provided rosters. Keep the JSON minified with double quotes.")

def roster_block(name,roster):
    rows=[f"{r} | {by_name[r]['position']} | {by_name[r]['projected_points']:.1f}" for r in roster]
    return f"{name} Roster (name | pos | proj):\n"+"\n".join(rows)

def render_prompt(variant,teamA,teamB):
    A=roster_block('Team A',teamA);B=roster_block('Team B',teamB)
    
    if variant=='tom':
        g=("You are the manager of Team A. Your goal is to improve Team A by proposing a mutually beneficial trade with Team B that Team B is likely to accept. Consider what team B needs or might accept. You may offer any players from your own team in exchange for players on Team B's roster. Carefully analyze both teams rosters, including positional strengths and weaknesses. Your trade should consider Team B's perspective which includes their positional needs and likely willingness to accept the trade and also be fair and realistic. Your task is to propose a trade and clearly state which player(s) each team would give and receive. Think step by step through your analysis. After, write a detailed explanation on why the trade benefits both teams, what needs each team is addressing and why Team B is likely to accept the trade.")
    else: 
        g=("You are the manager of Team A in a fantasy football league. Your goal is to improve your team by proposing a trade with team B. Review both rosters and propose a trade that accomplishes your goal. Provide a detailed explanation for your trade.")
    
    c=("You may trade any number of players. Trade must not include names outside rosters. Return JSON only, no prose.")
    return f"{PROMPT_HEADER}\n\n{A}\n\n{B}\n\nTask: {g} {c}"

def write_prompts(scenarios):
    man=[]
    for s in scenarios:
        for v in PROMPT_VARIANTS:
            for m in MODELS:
                f = OUTDIR/'prompts'/f"sc{s['scenario_id']}_{v}_{m}.txt"
                if not f.exists():
                    f.write_text(render_prompt(v, s['teamA'], s['teamB']))
                man.append({'scenario_id': s['scenario_id'], 'variant': v, 'model': m, 'prompt_path': str(f)})
    (OUTDIR/'prompts_manifest.json').write_text(json.dumps(man, indent=2))
    (OUTDIR/'scenarios.json').write_text(json.dumps(scenarios, indent=2))
    return man

MODEL_MAP={'gpt-4o':'gpt-4o','claude-3.5-sonnet':'claude-3-5-sonnet-20240620','gemini-2.5-pro':'models/gemini-2.5-pro'}

def call_openai(prompt,model_id):
    try:
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return r.choices[0].message.content
    except Exception as e:
        return None

def call_anthropic(prompt,model_id):
    try:
        import anthropic
        c=anthropic.Anthropic()
        m=c.messages.create(model=model_id,max_tokens=1000,temperature=0,messages=[{"role":"user","content":prompt}])
        t=[]
        for p in m.content:
            if getattr(p,'type','')=='text': t.append(p.text)
        return "\n".join(t)
    except Exception:
        return None

def call_gemini(prompt,model_id):
    try:
        import google.generativeai as genai
        model = genai.GenerativeModel(model_id)
        r = model.generate_content(prompt)
        return r.text
    except Exception as e:
        return None

def call_model(model,prompt):
    if model=='gpt-4o': return call_openai(prompt,MODEL_MAP[model])
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

def calculate_ideal_vector():
    total = sum(COUNTS.values())
    order = ['QB','RB','WR','TE']
    return [COUNTS.get(pos, 0) / total for pos in order]

IDEAL_VEC = calculate_ideal_vector()

KEYWORDS = {'need','needs','believe','likely','accept','balance','fit','perspective','goal','willing','weak','strong'}

def tom_keyword_score(text):
    if not isinstance(text,str): return 0
    t = text.lower()
    return sum(1 for w in KEYWORDS if w in t)

def advanced_tom_analysis(text):
    """Analyze text for various ToM reasoning indicators"""
    if not isinstance(text, str):
        return {'keywords': 0, 'perspective_taking': 0, 'need_analysis': 0, 'mutual_consideration': 0, 'reasoning_depth': 0}
    
    t = text.lower()

    keywords = sum(1 for w in KEYWORDS if w in t)
    
    perspective_phrases = [
        'team b needs', 'team b wants', 'they need', 'they want', 'their team', 'from their perspective', 
        'team b would', 'helps team b', 'team b gets', 'team b receives', 'for team b', 'team b benefits',
        'they would benefit', 'gives team b', 'team b\'s needs', 'what team b needs', 'team b is looking for'
    ]
    perspective_taking = sum(1 for phrase in perspective_phrases if phrase in t)
    
    need_phrases = [
        'needs more', 'lacking', 'weak at', 'shortage', 'depth at', 'upgrade at', 'improve their',
        'needs help', 'could use', 'would benefit from', 'needs depth', 'thin at', 'missing',
        'needs an upgrade', 'better at', 'stronger at', 'addresses needs'
    ]
    need_analysis = sum(1 for phrase in need_phrases if phrase in t)
    
    mutual_phrases = [
        'both teams', 'mutually beneficial', 'helps both', 'win-win', 'each team gets', 'addresses both',
        'benefits both', 'both sides', 'fair trade', 'addresses needs for both', 'good for both',
        'both teams benefit', 'each team', 'satisfies both'
    ]
    mutual_consideration = sum(1 for phrase in mutual_phrases if phrase in t)
    
    depth_indicators = [
        'because', 'since', 'therefore', 'as a result', 'this would', 'which means', 'leading to',
        'due to', 'given that', 'considering', 'allows', 'enables', 'provides', 'creates',
        'resulting in', 'consequently', 'thus', 'hence'
    ]
    reasoning_depth = sum(1 for phrase in depth_indicators if phrase in t)
    
    return {
        'keywords': keywords,
        'perspective_taking': perspective_taking, 
        'need_analysis': need_analysis,
        'mutual_consideration': mutual_consideration,
        'reasoning_depth': reasoning_depth,
        'total_tom_score': keywords + perspective_taking + need_analysis + mutual_consideration + reasoning_depth
    }

def positional_value_score(player_name):
    """Calculate positional value based on scarcity and points"""
    if player_name not in by_name:
        return 0
    
    pos = by_name[player_name]['position']
    points = by_name[player_name]['projected_points']
    scarcity = {'QB': 1.0, 'RB': 1.2, 'WR': 1.1, 'TE': 1.3}
    return points * scarcity.get(pos, 1.0)

def trade_balance_score(offer_names, ask_names):
    """Calculate trade balance using positional value, not just raw points"""
    offer_value = sum(positional_value_score(n) for n in offer_names)
    ask_value = sum(positional_value_score(n) for n in ask_names)
    if offer_value == 0 and ask_value == 0:
        return 1.0, 0.0, 0.0
    elif offer_value == 0 or ask_value == 0:
        return 0.0, offer_value, ask_value
    
    ratio = min(offer_value, ask_value) / max(offer_value, ask_value)
    return ratio, offer_value, ask_value

def team_fit_improvement(original_roster, new_roster):
    """Calculate how much a team's positional balance improves"""
    original_fit = cosine(pos_vector(original_roster), IDEAL_VEC)
    new_fit = cosine(pos_vector(new_roster), IDEAL_VEC)
    return new_fit - original_fit

def mutual_benefit_score(teamA_before, teamA_after, teamB_before, teamB_after):
    """Calculate if both teams benefit from the trade"""
    points_A = team_points(teamA_after) - team_points(teamA_before)
    points_B = team_points(teamB_after) - team_points(teamB_before)
    
    fit_A = team_fit_improvement(teamA_before, teamA_after)
    fit_B = team_fit_improvement(teamB_before, teamB_after)
    
    benefit_A = points_A > 0 or fit_A > 0.001 or (points_A > -2 and fit_A > -0.01)
    benefit_B = points_B > 0 or fit_B > 0.001 or (points_B > -2 and fit_B > -0.01)
    
    return int(benefit_A and benefit_B), points_A, points_B, fit_A, fit_B

def additional_tom_metrics(reasoning_text):
    """Additional sophisticated ToM analysis metrics"""
    if not isinstance(reasoning_text, str):
        return {
            'team_a_mentions': 0, 'team_b_mentions': 0, 'team_balance': 0,
            'justification_score': 0, 'specific_benefits': 0
        }
    
    import re
    text_lower = reasoning_text.lower()
    
    # Team mention analysis
    team_a_count = len(re.findall(r'\bteam a\b', text_lower))
    team_b_count = len(re.findall(r'\bteam b\b', text_lower))
    total_mentions = team_a_count + team_b_count
    team_balance = min(team_a_count, team_b_count) / total_mentions if total_mentions > 0 else 0
    
    # Specific benefit analysis
    benefit_patterns = [
        r'gives?\s+team\s+[ab]\s+\w+',
        r'team\s+[ab]\s+gets?\s+\w+',
        r'improves?\s+team\s+[ab]',
        r'helps?\s+team\s+[ab]\s+with',
        r'addresses?\s+team\s+[ab]'
    ]
    specific_benefits = sum(len(re.findall(pattern, text_lower)) for pattern in benefit_patterns)
    
    # Justification sophistication
    justification_indicators = [
        'upgrade', 'improvement', 'depth', 'strength', 'weakness',
        'complements', 'fills gap', 'addresses need', 'balances'
    ]
    justification_score = sum(1 for indicator in justification_indicators if indicator in text_lower)
    
    return {
        'team_a_mentions': team_a_count,
        'team_b_mentions': team_b_count, 
        'team_balance': team_balance,
        'justification_score': justification_score,
        'specific_benefits': specific_benefits
    }

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
                
                applied = apply_trade(s['teamA'], s['teamB'], tr)
                if not applied:
                    continue
                A_new, B_new = applied
                
                trade_balance, offer_value, ask_value = trade_balance_score(
                    tr.get('offer_from_A',[]), tr.get('ask_from_B',[]))
                
                mutual_benefit, points_A, points_B, fit_A, fit_B = mutual_benefit_score(
                    s['teamA'], A_new, s['teamB'], B_new)
                reasoning_text = tr.get('reasoning', '')
                tom_analysis = advanced_tom_analysis(reasoning_text)
                additional_metrics = additional_tom_metrics(reasoning_text)

                rows.append({
                    'scenario_id': s['scenario_id'], 
                    'variant': v, 
                    'model': m,
                    'points_change_A': points_A,
                    'points_change_B': points_B, 
                    'fit_change_A': fit_A,
                    'fit_change_B': fit_B,
                    'mutual_benefit': mutual_benefit,
                    'trade_balance': trade_balance,
                    'offer_value': offer_value,
                    'ask_value': ask_value,
                    'tom_keywords': tom_analysis['keywords'],
                    'perspective_taking': tom_analysis['perspective_taking'],
                    'need_analysis': tom_analysis['need_analysis'], 
                    'mutual_consideration': tom_analysis['mutual_consideration'],
                    'reasoning_depth': tom_analysis['reasoning_depth'],
                    'total_tom_score': tom_analysis['total_tom_score'],
                    'reasoning_length': len(reasoning_text.split()) if reasoning_text else 0,
                    'team_a_mentions': additional_metrics['team_a_mentions'],
                    'team_b_mentions': additional_metrics['team_b_mentions'],
                    'team_balance': additional_metrics['team_balance'],
                    'justification_score': additional_metrics['justification_score'],
                    'specific_benefits': additional_metrics['specific_benefits']
                })
    return pd.DataFrame(rows)

if __name__=='__main__':
    sc=load_or_make_scenarios()
    man=write_prompts(sc)
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
## Frozen Lake
Šance na SLIPPERY 1/3. </br>
Prováděná ACTION RIGHT / LEFT - možné slippery UP / DOWN / prováděná akce, </br>
Prováděná ACTION UP / DOWN - možné slippery RIGHT / LEFT / prováděná akce. </br>

Problém velké mapy - delší cesta = delší čas pro naučení Q-TABLE.

Po implementaci Q-LEARNING, EPSILON GREEDY POLICY další navrhované penalizace při trénování:
1. velké množství řešených cest,
2. využívání prozkoumaných cest, pozor ale na SLIPPERY,
3. v případě random generované mapy může docházet k vygenerování mapy bez díry, nebo naopak s velkým počtem děr (více než 6), které negativně ovliňuje trénování. 


## Reinforcement learning
| Term | Explanation |
| ---- | ---- |
| ACTION | "mechanismy rozhodnutí (možné pohyby) které AGENT využije pro přecházení mezi stavy ENVIRONMENT", "akce provedená AGENTem (v určitém STATE) ovlivňuje ENVIRONMENT" |
| AGENT | "entita, která používá POLICY k maximalizaci očekávané REWARD" |
| BELLMAN EQUATION | "rovnice využívána k vypočtení hodnoty Q-VALUE pro jednotlivé EPISODy"|
| ENVIRONMENT | "obsahuje AGENTa a umožňuje agentovi pozorovat stav poskytnutého světa", "v případě kdy AGENT aplikuje ACTION, ENVIRONMENT přechází mezi stavy" |
| EPISODE | "každá (opakovaná) sequence AGENTa naučit se ENVIRONMENT"|
|  | "Sequence: STATE -> ACTION -> REWARD -> REPEATE TILL TERMINATION / FINAL STATE REACHED" |
| EXPLO**RA**TION | "zkoušení nových možností za účelem nalezení mnoha (nových) informací o možných maximálních REWARDs v ENVIRONMENT"|
| EXPLO**TA**TION | "využívání známých (naučených) informací z ENVIRONMENT k získání maximální REWARD" |
| EPSILON GREEDY POLICY | "POLICY, která definuje agentovy ACTION a uskutečňuje výpočet Q-VALUE", "v následujících EPOSODách snižuje hodnotu EPSILON, aby změnil chování z EXPLO**RA**TION na EXPLO**TA**TION"|
| EPSILON ε | "parametr pro rozhodování mezi EXPLO**RA**TION a EXPLO**TA**TION" |
| RANDOM POLICY | "náhodné provedení ACTION AGENTem" |
| GREEDY POLICY | "vždy vybere ACTION s největší REWARD" |
| POLICY | "pravděpodobnost rozhodování AGENTa mezi vykonání STATE a ACTION" "v Q-LEARNING rozhodování ovlivněno největší Q-VALUE v tabulce pro každý stav" |
| Q-FUNCTION | "funkce, která předpovídá očekávaný REWARD z provedené ACTION v daném STATE za dodržení platné POLICY" |
| Q-LEARNING | "algoritmus, který umožňuje AGENTovi naučit se optimální Q-FUNCTION pomocí BELLMAN EQUATION" |
| Q-TABLE | "tabulka obsahuje nejlepší hodnoty REWARD za konkrétní provedené ACTION v ENVIRONMENT"|
| Q-VALUE | "číselná hodnota, která reprezentuje specifickou ACTION v rámci určitého STATE"|
| REWARD | "číselná hodnota navrácena ENVIRONMENTem po provedení ACTION AGENTEm v daném STATE", "vyjadřuje přechod z jednoho stavu do druhého v důsledku ACTION" |
| STATE | "hodnota, která popisuje aktuální konfiguraci ENVIRONMENT", "aktuální pozice AGENTa v ENVIRONMENTu využívána AGENTem pro volbu nejlepší ACTION" |

References: [Google Machine Learning Glossary], [Machine Learning Q-LEARNING terms].

   [Google Machine Learning Glossary]: <https://developers.google.com/machine-learning/glossary/rl#markov_decision_process>
   [Machine Learning Q-LEARNING terms]:
   <https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-q-learning#:~:text=EXPLORE%20PROGRAM-,Important,-Terms%20in%20Q>


## Quality-Learning algorithm
![Q-L algorithm](https://images.datacamp.com/image/upload/v1666973295/Q_Learning_Process_134331efc1.png)


### Q-Table
Fáze testování je zakončena vytvořením Q-TABLE s Q-VALUES odpovídající maximální dosažitelné REWARD. AGENT se podle vypočtené Q-TABLE rozhoduje v ENVIRONMENT. Sloupce představují konečný počet ACTION, které AGENT může v ENVIRONMENTu provést. Řádky představují jednotlivé STATES, ve kterých se AGENT nachází (např. po přechodu ze STATE 1 do STATE 2 pomocí ACTION, nebo v případě SLIPPERY).

[![Q-table-explained.png](https://i.postimg.cc/X7rcrtrW/Q-table-explained.png)](https://postimg.cc/Xp0yhHK1)

Vysvětlení nalezení 1. možné cesty:
1. AGENT začíná na políčku STATE[0],
2. Podle Q-TABLE vybere první hodnotu z NULTÉHO řádků blížící se nejvíc REWARD 1 (tedy 0.7737),
3. AGENT vykoná ACTION down,
4. Na nové pozici STATE[4] podle Q-TABLE vybere první hodnotu ze ČTVRTÉHO řádku blížící se nejvíc REWARD 1 (tedy 0.7737),
5. AGENT vykoná ACTION down,
5. Proces opakuje dokud z Q-TABLE nevybere hodnotu, která jej dostane do cíle (tedy na čtrnáctém řádku ACTION right),
6. Výsledná cesta: 0 - 4 - 8 - 9 - 13 - 14 - 15.

Existuje více možných cest. Předpokládám, že postup vyhodnocování pro další cesty bude podobný. Myslím si, že všechny možné cesty by reprezentoval veškerými kombinacemi, protože z Q-TABLE lze často vybrat více než jedna možná cesta (fakt není ověřen).


### Bellman Equation
![Bellman-Equation](https://images.datacamp.com/image/upload/v1714741243/image_2b5b99673a.png) </br>
Zjednodušeně: </br>
Q(s,a) ←  Q(s,a) +  α *  (r + γ * maxQ(s',a') - Q(s,a))

Rovnice v kódu:
```sh
Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma *np.max(Qtable[new_state]) - Qtable[state][action]) 
```
[Detailed explanation with code example]

[Detailed explanation with code example]:
<https://github.com/Viddesh1/RL/blob/main/Mini_Project.ipynb>


### Hyperparameters
| Alias | Term | | Explanation |
|---|---|---|---|
| ALPHA  α | learning rate | hodnoty 0 až 1 | "jak rychle se AGENT učí" |
| GAMMA γ | discount factor | hodnoty 0 až 1 | "jak moc AGET zvažuje využití budoucích REWARDs"|
||||
| Q(s,a) | | | "očekávaná odměna za provedení ACTION (a) ze STATE (s)" |
| maxQ(s',a') | | | "označuje maximální možnou hodnotu (Q-VALUE), kterou lze získat v následujícím stavu s' při použití libovolné akce z následujícího stavu s'", "obdoba předpovědi budoucnosti"|


### "Optimalized" hyperparameters
**ALFA** - Doporučené hodnoty 0.1 až 0.5. Příliš vysoké hodnoty (blížící se 1) způsobují, špatné učení AGENTA. Staré informace (i ty námi považované za optimální) budou přemazávany novými.  </br>
**GAMMA** - Doporučené hodnoty 0.9 až 0.99. V případě kdy GAMMA = 1, AGENT vyhledává řešení podle budoucích cest. Pokud ale GAMMA = 0, AGENT vyhledává maximální okamžitou (aktuálně získatelnou) odměnu.

## Epsilon greedy
Strategie, podle které se AGENT rozhoduje, jestli bude provádět EXPLO**RA**RION nebo EXPLO**TA**TION v rámci ENVIRONMENTu. AGENT začíná s vysokou hodnotou EPSILON ε aby prováděl EXPLO**RA**TION. S každou dokončenou EPISODE dochází k zmenšení EPSILONu ε, aby AGENT začal upřednostňovat EXPLO**TA**TION z hodnot v Q-TABLE. 

**ROVNICE** </br>
ϵ<sub>t</sub> = ϵ<sub>0</sub> * (DECAY_RATE)<sup>t</sup>

| Term | Explanation | Equation |
|---|---|---|
| EXPLO**RA**TION | "AGET zvolí ACTION s největší hodnotou v Q-TABLE | 1 - ε|
| EXPLO**TA**TION | "AGENT zvolí náhodnou POLICY (náhodnou ACTION)" | ε |
| ϵ<sub>t</sub> | "hodnota EPSILON v jednotlivé EPISODe"  | - |
| ϵ<sub>0</sub> | "počáteční hodnota"  | - |
| DECAY_RATE | "hodnota, o kterou se v každé EPISODě snižuje hodnota EPSILON", "rychlost snižování EPSILONu ε" | - |
| 0.99 |"menší DECAY_RATE znamená rychlejší přechod z EXPLO**RA**RION AGENTa v ENVIRONMENTu k EXPLO**TA**TION"||
| 0.999 | "větší DECAY_RATE znamená pomalejší přechod z EXPLO**RA**TION k EXPLO**TA**TION", "výhodnější pro větší / složitější ENVIRONMENT"||
| t | "počet vykonávaných EPISODE" | - |

**OPTIMALIZOVANÉ PARAMETRY** </br>
ϵ<sub>0</sub> - Doporučené hodnoty 0.9 až 1. </br>
ϵ<sub>0</sub> = 1.0: AGENT bude na začátku čistě náhodně zkoumat různé ACTION.  </br>
ϵ<sub>0</sub> = 0.9: AGENT bude 90 % času zkoumat nové ACTION a 10 % času využívat naučené Q-VALUEs.  </br>

t - Počet epizod: 2000 až 5000 pro jednodušší problémy, 10 000+ pro složitější prostředí.

[viz. other parameters examples]

[viz. other parameters examples]:
<https://medium.com/@MaLiN2223/frozen-lake-with-rl-my-journey-9b048e396ac3#:~:text=in%20the%20article.-,Parameters,-Unless%20explicitly%20mentioned>

### Others penalization methods
1. Negative Reward for Undesired Actions
2. Cost-sensitive Q-learning
3. Softmax Exploration
4. Punishment-based Q-learning
5. Inverse Reinforcement Learning (IRL)

## Related projects
Projekty jsou seřazeny podle subjektivního hodnocení, jak jsou užitečné a které problémy jsou řešeny.
1. [Greed policy, trainning / evaluation model, vizualization (GIF)] doprovázeno podrobným popisem [zde]
2. [Original GYM solution, hyperparameters, vizialization (graphs), custom map size]
3. [Each RL method implemented, Exploration VS. Explotation setting, greedy exploration]
4. [Graph / path visualization]

 [Greed policy, trainning / evaluation model, vizualization (GIF)]:
 <https://colab.research.google.com/drive/1IdsdHZ9Q8pAhmPfX4hHed150IG4YtnaX?usp=sharing#scrollTo=Y1tWn0tycWZ1>
 [zde]:
 <https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial#:~:text=POWERED%20BY-,Frozen%20Lake%20Gym%20Environment%C2%A0,-We%20are%20going>
 [Original GYM solution, hyperparameters, vizialization (graphs), custom map size]:
 <https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/>
[Each RL method implemented, Exploration VS. Explotation setting, greedy exploration]:
<https://github.com/xscchoux/Reinforcement-Learning-Frozen-Lake-/blob/master/Assignment-1-Part-2.ipynb>
[Graph / path visualization]:
<https://gsverhoeven.github.io/post/frozenlake-qlearning-convergence/>

Další, inspirativní řešení?
https://github.com/FareedKhan-dev/Reinforcement-Learning-on-Frozen-Lake-v1-openAI-gym
https://github.com/martin-ueding/game-simulation-sandbox/blob/203f71ee972b85358757e55ffef096c9c261a2a0/src/game_simulation_sandbox/frozenlake/
https://github.com/IJKTech/FrozenLake/blob/master/frozen_lake.py
https://github.com/YZK-yzk/DQN_FrozenLake_v0/blob/master/DQN_FrozenLake-v0.py


## Useful for code
Random map generation code changes:
```sh
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make("FrozenLake-v1", is_slippery=params.is_slippery, 
      render_mode="rgb_array", desc=generate_random_map(size=map_size, 
      p=params.proba_frozen, seed=params.seed),)
```

Print Q-table code changes:
```sh
# pridani nasledujiciho volani na konec if __name__ == '__main__':
 print(q_table)
```

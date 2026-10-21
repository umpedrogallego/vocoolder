# Guia dos Controles — Abas **Grãos** e **Pitch**

> Explicação em linguagem simples, sem termos técnicos, do que cada controle do **Vocoolder** altera no som.

---

## Como ler este guia

O programa tem uma coleção de pedacinhos gravados de voz — os **grãos**. Quando você fala ou canta, ele responde com esses pedacinhos.

| Aba | Decide |
|:--|:--|
| **Grãos** | *Quais* pedacinhos são escolhidos e *como* são disparados |
| **Pitch** | Em que *altura* (grave/agudo) esses pedacinhos soam |

```
+--------------------------------------------------------------+
| IMPORTANTE: nada aqui altera a sua voz ao vivo.               |
| O quanto você ouve sua voz x os pedacinhos é definido pelo    |
| "Mix Voz/Grãos", na aba Voz.                                  |
+--------------------------------------------------------------+
```

---

## Aba Grãos

### Visão rápida

| Controle | Faixa | Padrão | Em uma frase |
|:--|:--:|:--:|:--|
| Densidade do Ricochete | 1 – 6 | **3** | Quantos pedacinhos disparam por vez |
| Contraste Tímbrico | 0 – 100% | **20%** | O quanto o som imita você ou o oposto de você |
| Resposta Fonética | 0 – 100% | **60%** | O quanto ele imita o "tipo" do seu som (limpo, soprado, chiado) |

---

### Densidade do Ricochete

Controla **quantos pedacinhos** o programa dispara a cada vez que você faz som. Pense numa bola quicando: 1 é um quique só; 6 é uma sequência rápida de seis quiques.

Cada pedacinho entra uns 40 milésimos de segundo depois do anterior, em cascata, e todos ficam mais baixinhos para o volume total não estourar.

| Ajuste | Como soa |
|:--|:--|
| **1–2** | Resposta limpa, quase um eco único. Dá para entender o que você falou. |
| **3–4** | Equilíbrio: presença sem perder o desenho da fala. |
| **5–6** | "Nuvem" densa, enxurrada de sílabas. Menos fala, mais textura percussiva. |

---

### Contraste Tímbrico

Define se os pedacinhos escolhidos devem **se parecer com o seu som** ou **ser o oposto dele**.

| Ajuste | Efeito |
|:--|:--|
| **0%** | Escolhe sempre o pedacinho mais parecido com o que você está fazendo. O efeito "dobra" sua voz, imita você, soa coeso e discreto. |
| **20%** (padrão) | Quase sempre parecido, com um leve desvio que evita repetição. |
| **100%** | Escolhe o pedacinho mais diferente possível. Texturas estranhas, ásperas, imprevisíveis. |

> Em valores altos ele também faz os pedacinhos da **mesma cascata** serem bem diferentes entre si — a resposta fica um "mosaico" e não um bloco uniforme.

---

### Resposta Fonética

"Fonética", aqui, significa **tipo de som**: limpo e cantado versus soprado, sussurrado, chiado (como "sss", "shh", "fff") ou rasgado.

| Ajuste | Efeito |
|:--|:--|
| **0%** | Ignora se você está sussurrando ou cantando; olha só o formato geral do som. |
| **60%** (padrão) | Sensação de "ele está me escutando de verdade", sem travar num único tipo de som. |
| **100%** | Obedece quase cegamente ao grau de ar/ruído. Ótimo para sussurros e consoantes, mas menos variado. |

Quanto maior o valor, mais ele procura pedacinhos com o **mesmo grau de ar/ruído** que você está produzindo: se você sopra, responde com sopros; se canta limpo, responde limpo; se faz chiado, responde com chiado.

---

## Aba Pitch

Tudo aqui é sobre a **altura** dos pedacinhos — e vale apenas para os grãos, não para a sua voz.

### Visão rápida

| Controle | Opções | Padrão | Em uma frase |
|:--|:--|:--:|:--|
| Modo | 4 opções | **Desvio microtonal** | Como a altura dos pedacinhos é decidida |
| Intervalo | 6 opções | **Quinta justa** | Distância usada (só no modo "Intervalo fixo") |
| Afinação | 2 opções | **24-EDO** | A "régua" que arredonda as alturas |
| Preservar Duração | 0 – 100% | **0%** | Mantém o comprimento original ao mudar a altura |

---

### Campo "Modo"

| Opção | O que faz | Quando usar |
|:--|:--|:--|
| **Desligado** | Pedacinhos tocam na altura em que foram gravados. | Ouvir a biblioteca "crua"; máxima previsibilidade. |
| **Desvio microtonal** | Imita as pequenas escorregadas de afinação da sua voz, em degraus minúsculos, e espalha as cópias da cascata com pequenos desvios entre si. Também responde ao volume: mais forte = puxa para cima. | Brilho trêmulo, "coro desafinadinho de propósito". O modo mais vivo e menos óbvio. |
| **Casar com a voz** | Ajusta cada pedacinho para a **sua nota atual** (até uma oitava de distância). | Reforço afinado, em uníssono, sem "nota errada". |
| **Intervalo fixo** | Afasta os pedacinhos uma distância fixa da altura original deles. **Não se ajusta à sua nota.** | Harmonia experimental: pode soar consonante ou áspero, dependendo do pedacinho sorteado. |

> Em **Casar com a voz**, se o programa não consegue identificar sua nota (sussurro, ruído, silêncio), ele simplesmente não mexe na altura.

---

### Campo "Intervalo"

Só tem efeito no modo **Intervalo fixo**. Os nomes são distâncias musicais; a coluna da direita traduz a sensação.

| Opção | Sensação |
|:--|:--|
| **Uníssono** | Mesma altura dos pedacinhos originais. Dobra o som sem subir nada. |
| **Quarta justa** | Sobe um pouco (de "dó" para "fá"). Distância agradável e estável. |
| **Trítone** | Sobe um pouco mais que a quarta. A mais tensa/estranha da lista — misteriosa ou incômoda de propósito. |
| **Quinta justa** (padrão) | Sobe mais um pouco (de "dó" para "sol"). A mais "redonda" depois da oitava: forte e consonante. |
| **Sétima menor** | Sobe bastante, quase uma oitava. Sabor mais "blues"/dramático. |
| **Oitava** | Dobra a altura — a mesma nota, uma região acima. O mais seguro e chamativo. |

---

### Campo "Afinação"

Define a "régua" usada para arredondar as alturas.

| Opção | Como funciona | Caráter |
|:--|:--|:--|
| **24-EDO** (padrão) | Divide a oitava em 24 passos iguais (os "quartos de tom"). Régua de precisão média. | Moderno, levemente "fora do lugar", colorido meio eletrônico. |
| **JI** | Usa as relações "naturais" da física do som (as que aparecem quando uma corda vibra). | Mais puro e estável, sobretudo em sons longos. Alguns intervalos ganham tempero próprio (a sétima soa mais "larga"). |

> Resumo: **24-EDO** = passos iguais e previsíveis; **JI** = relações naturais, mais vivas e com caráter próprio.

---

### Preservar Duração

Ao afinar um pedacinho para cima ou para baixo, ele naturalmente fica **mais curto ou mais longo** — como um disco de vinil acelerado ou uma fita passada devagar (o efeito "chipmunk").

| Ajuste | Comportamento | Resultado sonoro |
|:--|:--|:--|
| **0%** (padrão) | Mudar a altura muda também a velocidade e o comprimento. | Cru, analógico, coerente. Agudos ficam curtinhos; graves ficam arrastados. |
| **No meio** | Mistura parcial das duas coisas. | Meio-termo; o controle anda em passos de 5%. |
| **100%** | O programa "conserta" o comprimento e mantém a duração original, mudando só a altura. | Controlado e "produzido", mas pode ganhar um leve aspecto metálico/vidrado. |

> Dica: quanto mais você usa os modos de pitch (especialmente **Oitava**), mais vale subir esse controle — senão os pedacinhos agudos encurtam demais e somem.

---

## Combinações de bolso

| Objetivo | Ajustes sugeridos |
|:--|:--|
| **Inteligível e próximo da sua voz** | Contraste 10–20% · Fonética ~60% · Densidade 2–3 · Modo "Casar com a voz" ou "Desligado" |
| **Textura estranha e imprevisível** | Contraste 70–100% · Densidade 5–6 · Modo "Desvio microtonal" ou "Intervalo fixo" com Trítone/Sétima |
| **Coro etéreo afinado** | Modo "Casar com a voz" · Afinação JI · Densidade 4–6 · Preservar Duração 40–60% |
| **Notas longas e drones** | Preservar Duração 70–100% · Afinação JI |

---

*Guia baseado nos controles reais das abas Grãos e Pitch do Vocoolder (valores padrão e limites conferidos no código do motor e da interface).*

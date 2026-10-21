# Vocoolder - Sistema Reativo de Performance Vocal

## Visão Geral

O **Vocoolder** processa a voz do microfone em tempo real e mistura grãos do
corpus (`amostras/`) com deslocamento microtonal (24-EDO / Just Intonation),
ricochete fônico, resposta fonética e contraste tímbrico. O pitch pode ser
desligado, microtonal, casado com a voz ou fixo, com varispeed/WSOLA.

O caminho em tempo real é **leve**: `numpy`, `scipy` e `sounddevice`.
`torch`, `torchaudio`, `librosa`, `numba` e `sklearn` **não** são usados no
loop ao vivo — isso elimina os travamentos e o consumo excessivo de CPU/memória.

A biblioteca pode ser maior que o conjunto ativo. O cache guarda todos os
grãos da biblioteca; o motor carrega apenas os **grãos ativos** escolhidos
(por fonte e por limite), liberando o áudio dos demais. Presets versionados
reproduzem voz, grãos, pitch, saída e biblioteca, e a sessão pode ser gravada
de forma assíncrona em `recordings/`.

## Requisitos

- Windows 10, Python 3.10
- `customtkinter`, `sounddevice`, `soundfile`, `numpy`, `scipy`

`librosa`, `torch` e `torchaudio` continuam instalados no ambiente, mas o
Vocoolder não depende deles para funcionar.

## Como Executar

### Interface gráfica

```powershell
python main.py
```

Controles:

| Controle | Função |
|---|---|
| **Iniciar Vocoolder / Parar** | Abre/fecha os streams em thread separada (a UI não trava) |
| **Densidade do Ricochete (1–6)** | Quantos grãos entram em cascata por evento |
| **Contraste Tímbrico (0–100%)** | 0% = grãos mais próximos; 100% = maior contraste, sem trocar de registro |
| **Resposta Fonética (0–100%)** | 0% = só timbre; 100% = segue o percentil de ruidosidade da voz |
| **Mix Voz/Grãos (0–100%)** | 0% = só grãos; 100% = só voz; padrão 30% |
| **Pitch (aba Pitch)** | Desligado, Desvio microtonal (padrão), Casar com a voz ou Intervalo fixo |
| **Intervalo fixo** | Uníssono, quarta, trítone, quinta, sétima menor ou oitava (sem sétima maior) |
| **Afinação** | 24-EDO (passos de 50 cents) ou JI (intervalos justos) |
| **Preservar Duração (0–100%)** | 0% = varispeed; 100% = WSOLA restaura a duração; intermediário restaura parcialmente |
| **Gate de Silêncio (RMS)** | Limiar de energia; ajustado automaticamente ao iniciar |
| **Calibrar Gate (ficar em silêncio)** | Recalibra o ruído ambiente com o motor rodando |
| **Grãos ativos (10–1.800)** | Tamanho do conjunto live; a biblioteca em disco pode ser maior |
| **Fontes...** | Marca quais amostras de `amostras/` alimentam o conjunto ativo |
| **Preset** | Salva, carrega e exclui configurações em JSON versionado |
| **Gravar sessão** | Grava a saída pós-limiter e um log de eventos em JSONL |
| **Testar Áudio** | Grava o microfone e toca dois tons na saída escolhida |
| **Medidor** | Nível de entrada em dBFS e pico de saída |

A área de controles é dividida em abas **Voz / Grãos / Pitch / Saída / Sessão**.
A lista de dispositivos mostra o **padrão do Windows** (incluindo saídas HDMI).
A sugestão exibida no log é apenas informativa — nada é trocado à força.

Ao alterar **fontes** ou **grãos ativos** com o motor rodando, pare e inicie
novamente para reconstruir o conjunto live. Presets que mudam biblioteca ou
dispositivos também exigem o motor parado; a troca é validada antes de aplicar
qualquer controle, para não deixar a configuração pela metade.

### Diagnóstico e testes (linha de comando)

```powershell
# Lista dispositivos (marca o padrão do Windows)
python main.py --list-devices

# Grava 2 s do microfone e toca A4/E5 na saída padrão
python main.py --diagnose

# Medidor de entrada em tempo real
python main.py --meter --seconds 15

# Escolher dispositivos específicos (veja os índices em --list-devices)
python main.py --diagnose --input-device 1 --output-device 12

# Gravar voz para testar o pipeline offline
python main.py --record 5
# -> voice_test.wav

# Auditar a extração dos grãos
python main.py --audit-corpus

# Exportar grãos para escuta (corpus_debug/)
python main.py --dump-grains 12

# Selftest: corpus + seleção + fonética + mix + pitch + WSOLA + render
python main.py --selftest

# Render offline de um arquivo (não é tempo real)
python main.py --render "amostras\05 Borboleta.mp3" -o render_teste.wav

# Variações de resposta fonética e contraste tímbrico
python main.py --render "voice-test\voice-test-01.wav" -o phon0.wav --phonetic 0 --contrast 0.2
python main.py --render "voice-test\voice-test-01.wav" -o phon100.wav --phonetic 1 --contrast 0.2
python main.py --render "voice-test\voice-test-01.wav" -o contrast100.wav --phonetic 0.6 --contrast 1.0

# Mix voz/grãos: 0 = só grãos, 1 = só voz
python main.py --render "voice-test\voice-test-01.wav" -o so_graos.wav --mix 0
python main.py --render "voice-test\voice-test-01.wav" -o so_voz.wav --mix 1

# Pitch: quinta fixa, 24-EDO, varispeed (0%) e WSOLA (100%)
python main.py --render "voice-test\voice-test-01.wav" -o quinta_tape.wav --pitch-mode interval --interval fifth --preserve 0
python main.py --render "voice-test\voice-test-01.wav" -o quinta_wsola.wav --pitch-mode interval --interval fifth --preserve 1

# Casar o grão com o f0 da voz, ou usar afinação justa
python main.py --render "voice-test\voice-test-01.wav" -o match.wav --pitch-mode match
python main.py --render "voice-test\voice-test-01.wav" -o quinta_ji.wav --pitch-mode interval --interval fifth --tuning ji

# Conjunto ativo menor e restrito a uma fonte (também vale para renders)
python main.py --render "voice-test\voice-test-01.wav" -o ativo100.wav --active-grains 100 --sources "01 Goticulas.mp3"

# Reindexar uma biblioteca maior (o cache existente só é substituído
# quando você usa --force-reindex; veja "Adicionar novas amostras")
python main.py --force-reindex --library-grains 2400 --per-file-cap 600 --audit-corpus
```

Todos esses modos usam apenas as dependências necessárias. O `--diagnose`,
`--meter`, `--list-devices` e `--record` não carregam o engine. O `--record`
grava apenas o microfone seco; para gravar a performance processada use o
botão **Gravar sessão** na GUI.

## Funcionamento Interno

### Corpus

- Leitura de WAV/MP3/FLAC/OGG com `soundfile`.
- Detecção de onsets por fluxo de energia em `numpy`.
- Multi-escala **60/120/180/350 ms**, hop de 70 ms, por onsets + grade regular.
- Padrão de **400 grãos/arquivo** e **1.800 no total**. Esses valores podem ser
  ampliados na reindexação com `--per-file-cap` e `--library-grains`.
- A biblioteca é o que fica no cache; o **conjunto ativo** (10–1.800 grãos)
  é escolhido por fonte e por amostragem estratificada de classe × escala.
  O áudio dos grãos inativos é liberado da RAM, mas continua no cache em disco.
- Cotas da paleta: **38% transiente/curto, 24% tonal/sustentado, 38% ruidoso/fricativo**.
- Seleção por cotas de classe × escala, cobertura temporal (bins de 5 s),
  diversidade em features (FPS) e mínimo por arquivo.
- Filtro de silêncio: RMS < 0,004 é descartado; o `maja-study` (54% não-silêncio)
  não domina o corpus com pausas. Sem remoção de ruído intencional.
- Descritores por grão: ZCR, centroid, flatness, HF 4–8 kHz, RMS, pico,
  f0, confiança de vozeamento e `noisy` (0–1).
- Normalização de pico para 0,7; áudio em `float16` no cache.
- Cache: `corpus_cache/corpus_data.pkl` (~29 MB com 8 arquivos / ~8 min).
- Assinatura por nome/tamanho/mtime: se uma amostra mudar, o corpus reindexa
  sozinho. Use `--force-reindex` para forçar.

### Features e busca

- 13 MFCCs (`numpy` + `scipy.fft.dct`) + centroid espectral + ZCR (15 dims).
- Normalização por média/desvio e similaridade **cosseno** em `numpy`.
- **Resposta Fonética**: percentil de `noisy` da consulta no corpus;
  0% = timbre puro, 100% = segue a ruidosidade da voz. O score combina
  ZCR 35%, flatness 35%, HF 4–8 kHz 20% e (1 − vozeamento) 10%.
- **Contraste Tímbrico**: 0% = vizinho mais próximo; 100% = maior distância
  de timbre. Penaliza saltos de registro (vozeado ↔ não vozeado e ±12 semitons)
  e adiciona diversidade entre os grãos da mesma cascata.
- Cascata: cada passo escolhe um grão **diferente** do anterior, com
  penalidade para os últimos 12 grãos usados.

### Biblioteca em dois níveis

- **Biblioteca**: todos os grãos do cache `corpus_cache/corpus_data.pkl`.
- **Conjunto ativo**: subconjunto carregado para seleção e playback, limitado
  pelo slider **Grãos ativos** (10–1.800) e pelo filtro de **Fontes**.
- A seleção ativa é determinística e preserva as cotas de classe (38/24/38) e
  a distribuição de escalas; dentro de cada grupo, espalha os grãos por
  arquivo e posição para não concentrar uma única fonte.
- O cache continua completo em disco. Ao montar o conjunto ativo, o áudio dos
  grãos não usados é substituído por um array vazio e sai da RAM. Para trocar
  fontes/limite, pare e inicie o motor: o cache é recarregado e um novo
  subconjunto é montado.
- O limite global de 1.800 grãos vale para o conjunto ativo. Para uma
  biblioteca maior, reindexe com `--library-grains` e `--per-file-cap`;
  o slider continua limitado a 1.800 no live.

### Presets

- Arquivos JSON em `presets/`, com `version`, `name`, `saved_at` e `sections`.
- Seções opcionais: **Biblioteca**, **Voz**, **Grãos**, **Pitch** e **Saída**.
  Só as seções marcadas são salvas e aplicadas.
- Fontes são gravadas como nomes relativos a `amostras/` (sem caminho
  absoluto). Dispositivos são gravados por **nome + hostapi**, nunca por índice.
- A gravação do JSON é atômica (`arquivo temporário` + `os.replace`). Ao
  carregar, todos os dispositivos são resolvidos e as regras de reinício são
  validadas antes de qualquer alteração; se algo falhar, nada é aplicado.
- Presets com biblioteca ou troca de dispositivo exigem o motor parado.
  Presets só com voz/grãos/pitch/gate podem ser aplicados ao vivo.

### Gravação de sessão

O botão **Gravar sessão** cria dois arquivos com o mesmo carimbo de tempo:

```text
recordings/session_AAAAMMDD_HHMMSS_mmm.wav   # saída pós-limiter
recordings/session_AAAAMMDD_HHMMSS_mmm.jsonl # eventos e grãos usados
```

O WAV usa a taxa e os canais negociados na saída (PCM 24 bits). O JSONL
começa com `session_start` e depois traz um objeto `event` por evento, com
tempo, posição em amostras, parâmetros, consulta de voz e a lista de grãos
(`grain_id`, arquivo, classe, escala, `start_s`, `onset`, `cents` etc.).
A gravação não bloqueia o callback; a fila tem 256 blocos (~3 s a 44,1 kHz).

### Áudio em tempo real

- `InputStream` e `OutputStream` separados (aceita mono/estéreo).
- O playback reutiliza o áudio `float16` do corpus; cada grão é convertido
  para `float32` só durante o processamento do evento.
- Callbacks só copiam buffers e atualizam níveis; o processamento roda em
  thread própria.
- Filas **não bloqueantes**; se algo atrasar, o bloco é descartado.
- Gate RMS com calibração automática: mede a **mediana** do ruído ambiente,
  aplica `gate = max(ruído × 3,5, 0,006)` e usa histerese (2 blocos para abrir,
  3 para segurar), evitando eventos por ruído de fundo.
- Pitch em quatro modos: **Desligado**, **Desvio microtonal**, **Casar com a
  voz** e **Intervalo fixo**. O desvio usa o f0 suavizado + energia; o match
  alinha o f0 do grão ao da voz, limitado a ±1.200 cents e exige confiança
  de vozeamento > 0,3. O f0 da voz passa por mediana de 3 eventos e histerese
  de 20 cents.
- Intervalos fixos: uníssono, quarta, trítone, quinta, sétima menor e oitava
  (sem sétima maior). Em JI, os intervalos usam razões justas (4/3, 3/2, 7/4…).
- Pitch por varispeed (`scipy.signal.resample_poly`) com quantização 24-EDO
  (50 cents) ou JI, conforme `microtonal_tables.py`.
- **Preservar Duração**: 0% = varispeed puro; 100% = WSOLA restaura o
  comprimento; valores intermediários restauram parcialmente. Não há
  preservação explícita de formantes.
- Fade Hann de 6 ms aplicado depois do pitch/WSOLA, nas bordas de cada grão.
- Mix único voz/grãos (padrão 30%): 0% = só grãos, 100% = só voz,
  com crossfade de potência constante (seno/cosseno) e suavização.
  Em 100% os eventos granulares são desativados; em 0% a voz seca é mutada.
- Cascata de até 6 grãos, 40 ms entre eles.
- Limiter suave com teto em **−1 dBFS** (0,891), transparente abaixo de 0,70.
- Watchdog: eventos acima de 80 ms são registrados no log; o selftest informa
  também o tempo máximo observado.
- Gravação de sessão assíncrona: o bloco pós-limiter entra em uma fila
  limitada e uma thread dedicada escreve o WAV; o callback de áudio nunca
  toca no disco. Se a fila encher, os blocos excedentes são contados e
  descartados, sem bloquear o playback.

### Defaults e validação final

- Densidade **3**, contraste **20%**, resposta fonética **60%**, mix **30%**.
- Pitch **Desvio microtonal**, intervalo **quinta**, afinação **24-EDO**,
  preservar duração **0%**.
- Conjunto ativo padrão: **1.800 grãos** (todas as fontes).
- Gate automático ao iniciar; saída padrão do Windows respeitada.
- Cache da biblioteca: **29,3 MB** com 1.800 grãos.
- RAM adicional de playback + cache de pitch: **~17 MB** a 44,1 kHz
  (áudio float16 reutiliza o corpus) e **~48 MB** a 48 kHz (limite de 80 MB).
- Evento mais lento medido com densidade 6 + WSOLA 100%:
  **~55 ms**; watchdog **0** eventos acima de 80 ms.
- `python main.py --selftest`: **PASS** (corpus, seleção, fonética, mix,
  quatro modos de pitch, intervalos, match de f0 e WSOLA).
- Conjunto ativo validado com 10, 100 e 1.800 grãos; a seleção preserva as
  cotas de classe e a distribuição de escalas.
- Gravação de sessão validada com WAV PCM 24 bits e JSONL contendo eventos,
  grãos e parâmetros.
- Saída de teste atual: **#12 Fones de ouvido, 44,1 kHz**; nada é trocado
  automaticamente.

### `src/microtonal_tables.py`

- `cents_to_semitones(cents)`
- `quantize_to_microtone(cents, mode='24edo')` — desvios em passos de 50¢
- `quantize_absolute_cents(cents, mode='24edo' | 'ji')` — intervalos absolutos
- JI de desvio usa a tabela de harmônicos (H7, H11, H13, H17)
- Tabela 24-EDO (passos de 50 cents)
- Intervalos JI absolutos (4/3 = 498,04¢, 3/2 = 701,96¢, 7/4 = 968,83¢…)
- Desvios JI harmônicos: H7 −31¢, H11 −49¢, H13 −41¢, H17 +5¢
- Constantes de formantes (F0–F3), sibilância em 4000 Hz

## Adicionar novas amostras

Formatos aceitos: **WAV, MP3, FLAC, OGG, M4A, AIFF/AIF**. Basta copiar o
arquivo para `amostras/`; o nome pode ter espaços e acentos.

1. **Copie a amostra** para `amostras/`.
2. **Reindexe ou deixe o programa detectar.** A assinatura usa nome, tamanho e
   data de modificação. Se o cache existente já foi criado e a assinatura
   mudou, a próxima execução reindexa sozinha. Para garantir que a nova fonte
   entre mesmo com a biblioteca cheia, use uma reindexação explícita:

   ```powershell
   # mantém o padrão de 1.800 grãos / 400 por arquivo
   python main.py --force-reindex --audit-corpus

   # ou amplia a biblioteca (o cache existente só é trocado com --force-reindex)
   python main.py --force-reindex --library-grains 2400 --per-file-cap 600 --audit-corpus
   ```

3. **Escolha o conjunto ativo.** Abra `python main.py`, vá em **Sessão >
   Fontes...**, marque a nova amostra (ou deixe todas) e ajuste **Grãos ativos**.
   Pare e inicie o motor depois de alterar fontes/limite.
4. **Salve um preset** em **Sessão** com a seção **Biblioteca** marcada. O
   preset guarda os nomes relativos das fontes, então funciona desde que os
   arquivos continuem em `amostras/`.

Observações:

- `--library-grains` e `--per-file-cap` só têm efeito na reindexação. Sem
  `--force-reindex`, um cache antigo é carregado; se ele for menor que o
  orçamento solicitado, o programa avisa para reindexar.
- O conjunto ativo continua limitado a 1.800 grãos por segurança de CPU/RAM,
  mesmo que a biblioteca tenha mais. Para um live mais leve, use 100–400.
- Orçamento aproximado do cache: **~16 KB por grão** (29 MB para 1.800).
  Uma biblioteca de 2.400 grãos fica em torno de 40 MB; o áudio do conjunto
  ativo é a maior parcela da RAM de playback.
- Amostras muito longas são suportadas, mas o indexador percorre o arquivo
  inteiro; prefira trechos de 1–3 minutos para reindexar mais rápido.
- Se a fonte não aparecer, confirme a extensão, rode `--audit-corpus` e veja
  se o arquivo foi listado. Fontes ausentes em um preset geram aviso e o
  conjunto ativo cai para as fontes disponíveis.

## Solução de Problemas

| Problema | Solução |
|---|---|
| Nenhum som | Rode `python main.py --diagnose`; confirme que ouviu os dois tons |
| Microfone muito baixo | `python main.py --meter --seconds 15`; ajuste o volume em Windows > Som > Entrada |
| Microfone saturando (pico 0 dBFS) | Reduza o ganho do microfone no Windows (picos ideais: −12 a −6 dBFS) |
| Gate não abre | Use **Calibrar Gate** em silêncio ou reduza o slider |
| Microfonia | Use fones de ouvido; a voz seca é monitorada na saída |
| Saída HDMI | É suportada normalmente; o dispositivo padrão do Windows é respeitado |
| Eventos acima de 80 ms / estalos | Reduza a densidade ou **Preservar Duração**; prefira saída 44,1 kHz e feche apps pesados |
| Pitch artificial ou metálico | Use **Preservar Duração 0%** (varispeed) ou reduza o valor; experimente 24-EDO |
| Corpus desatualizado | `python main.py --force-reindex --audit-corpus` |
| Quero ouvir os grãos | `python main.py --dump-grains 12` e abra `corpus_debug/montage.wav` |
| Nova amostra não entra no live | Reindexe com `--force-reindex --library-grains ...`; marque a fonte em **Sessão > Fontes** e reinicie o motor |
| Preset não encontra dispositivo | O preset guarda nome/hostapi; reconecte o dispositivo ou selecione outro e salve novamente |
| Gravação com perdas na fila | Use um disco mais rápido, reduza a taxa/canais ou pare outros processos; o playback continua intacto |
| Quero trocar o conjunto ativo | Pare o motor, ajuste **Grãos ativos**/**Fontes** e inicie novamente |

## Estrutura

```
vocoolder/
├── amostras/                 # áudios de referência
├── corpus_cache/             # biblioteca completa em cache (gerado automaticamente)
├── corpus_debug/             # grãos exportados para escuta (opcional)
├── metadata_csv/             # identidade/descritores dos grãos e resumo
├── presets/                  # presets JSON versionados
├── recordings/               # sessões gravadas (WAV + JSONL)
├── renders/                  # renders offline de validação
├── main.py                   # CLI + entrada da GUI
├── export_metadata_csv.py    # exporta metadados do cache para CSV
├── src/
│   ├── engine.py             # corpus, biblioteca ativa, features, streaming
│   ├── gui.py                # interface customtkinter
│   ├── presets.py            # schema e persistência atômica de presets
│   ├── session.py            # gravador assíncrono e log JSONL
│   ├── audio_diag.py         # diagnóstico, medidor, gravação seca
│   └── microtonal_tables.py  # tabelas microtonais e formantes
├── README.md
└── AGENTS.md
```

## Licença

Projeto de performance vocal experimental. Consulte `AGENTS.md` para as
diretrizes de desenvolvimento.

# Vocoolder - Diretrizes de Desenvolvimento

## Ambiente e Plataforma
- Sistema Operacional: Windows 10
- Python: 3.10
- Aceleração: NVIDIA CUDA disponível via PyTorch (`torch.cuda.is_available()`)

## Dependências Já Instaladas (Não reinstalar via pip)
- Áudio & DSP: `librosa`, `sounddevice`, `soundfile`, `scipy`, `numpy`, `torchaudio`
- Machine Learning: `scikit-learn` (k-NN)
- Interface: `customtkinter`

## Microtonal e Formant Tables
- Tabela `src/microtonal_tables.py` agora fornece:
  - Conversão cents ⇄ semitones via `cents_to_semitones()` para `torchaudio.transforms.PitchShift`
  - Quantização contínua para graus microtonais via `quantize_to_microtone(mode='24edo')`
  - Tabela 24-EDO (quartos de tom) com passos de 50 cents
  - Deviations de Just Intonation em relação ao temperamento igual ( Série Harmônica: harmônico 7 = -31¢, harmônico 11 = -49¢, harmônico 13 = -41¢, harmônico 17 = +5¢)
  - Constantes de formantes vocais (F0, F1, F2, F3) e zonas de atrito/sibilância

## Pipeline Acústico e Streaming
1. **Streaming contínuo sem travamento**: Use sempre `threading.Thread(daemon=True)` para rotinas de captura e processamento de áudio via `sounddevice`. A thread principal do `customtkinter` deve permanecer livre para atualização da interface.
2. **Janelamento de grains**: Sempre aplique janela Hann / fade in-out de 5ms a 10ms nas bordas dos grains de áudio para eliminar cliques e descontinuidades de fase.
3. **Identificação de dispositivo**: O código deve inicializar identificando automaticamente os canais padrão de entrada (microfone) e saída (alto-falantes/fones) de forma tolerante a mono/estéreo usando `sounddevice`.
4. **Microtonal pitch handling**: Varções contínuas do microfone devem passar por `quantize_to_microtone()` antes de alimentar o engine, garantindo que desvios sutis sejam mapeados para degraus microtonais definidos (24-EDO ou outro modo).
5. **Pitch shifting**: Use `cents_to_semitones()` para converter desvios em semitons float, alimentando diretamente os transforms de pitch do `torchaudio`.

## Simplicidade
Priorize código direto, modular e autocontido (máximo de 2 a 3 arquivos claros: `main.py`, `engine.py`, `gui.py`).

## Engenharia de Áudio - Regras Adicionais
- Nunca bloqueie a thread principal do GUI com processamento de áudio em tempo real.
- Use `torchaudio.transforms.PitchShift` com saída de `cents_to_semitones()` para transposições microtonais.
- Valores de ZCR (Zero Crossing Rate) acima de 3500 Hz indicam região de atrito/fricativas.
- Sibilância deve ser detectada próximo a 4000 Hz (ver `src/microtonal_tables.py` - constante `SIBILANCE`).
- Formantes F1 (80-750 Hz) e F2 (1200-3200 Hz) devem ser monitorados para classificação de vogais em tempo real.
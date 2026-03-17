Lab p1-04: O Transformer Completo "From Scratch"

Este repositório contém o código fonte correspondente ao Laboratório 04, focado na construção da arquitetura completa de um Transformer (Encoder-Decoder) construído a partir do zero utilizando PyTorch.

Neste projeto, atuei na integração dos módulos matemáticos desenvolvidos nos laboratórios anteriores, garantindo o fluxo correto de tensores ao longo de toda a rede neural para realizar um teste de tradução fim-a-fim.

📌 Arquitetura e Implementação

O projeto foi estruturado para cumprir rigorosamente as quatro tarefas propostas:

Refatoração e Integração (Tarefa 1): Adaptação dos motores legados para aceitar tensores dinâmicos do PyTorch.

Pilha do Encoder (Tarefa 2): Implementação do EncoderBlock, garantindo o fluxo bidirecional de Self-Attention.

Pilha do Decoder (Tarefa 3): Implementação do DecoderBlock com Causal Masking e ponte de Cross-Attention.

Loop de Inferência (Tarefa 4): Construção do laço auto-regressivo para tradução da frase "thinking machines".

🚀 Como Executar

Basta executar o script principal no terminal:

python transformer.py


⚖️ Declaração de Integridade Académica

"Partes geradas/complementadas com IA, revisadas por Marcus David Nascimento de Sá"

Nota de Desenvolvimento: Ferramentas de IA foram utilizadas para brainstorming e templates de código, conforme as diretrizes da disciplina. A lógica matemática e a validação do modelo foram realizadas integralmente por mim.

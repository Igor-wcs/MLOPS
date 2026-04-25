# 🛡️ Mapeamento de Ameaças (OWASP Top 10) - API Previsão PETR4

Este documento mapeia as 5 principais ameaças de segurança relevantes para a nossa arquitetura (baseado no OWASP Top 10 API Security 2023) e as mitigações que implementamos.

## 1. API3:2023 - Broken Object Property Level Authorization (Injeção de Dados)
* **Ameaça:** Atacantes podem tentar enviar códigos maliciosos, textos ou dados formatados incorretamente no corpo do JSON para tentar quebrar a API ou a rede neural (Data Poisoning/Injection).
* **Mitigação:** Utilizamos o framework **Pydantic** no FastAPI. A classe `DadosEntrada` exige estritamente um formato `List[float]`. Qualquer envio de texto, valores nulos ou formatos não numéricos é barrado automaticamente na porta de entrada (Erro 422 - Unprocessable Entity), blindando o PyTorch.

## 2. API4:2023 - Unrestricted Resource Consumption (DoS / Negação de Serviço)
* **Ameaça:** Envio intencional de requisições com payloads gigantescos (ex: uma lista contendo 1 milhão de preços) para estourar a memória RAM do servidor (OOM) e causar instabilidade.
* **Mitigação:** Implementamos uma validação estrita de Length (Tamanho) na rota de predição. A API aceita **exatamente 30 itens** (`if len(entrada.precos) != 30:`). Arrays maiores ou menores são imediatamente rejeitados (Erro 400), garantindo tempo de inferência constante e protegendo a infraestrutura.

## 3. API7:2023 - Security Misconfiguration (Exposição de Dados via Traceback)
* **Ameaça:** Erros internos (500) não tratados podem retornar mensagens de sistema (Tracebacks) que revelam a estrutura de pastas do servidor, versões de bibliotecas vulneráveis ou a lógica do negócio para o atacante.
* **Mitigação:** O FastAPI intercepta falhas (como problemas ao carregar o artefato do MLflow) e encapsula os erros em respostas seguras usando `HTTPException`. O Traceback original é ocultado do usuário final, retornando apenas mensagens controladas (ex: "Modelo não carregado no servidor").

## 4. API9:2023 - Improper Inventory Management (APIs Sombra)
* **Ameaça:** Endpoints legados, esquecidos ou mal documentados tornam-se alvos fáceis, pois a equipe de segurança e monitoramento não sabe da existência deles ("Shadow APIs").
* **Mitigação:** O FastAPI gera automaticamente a documentação OpenAPI/Swagger "viva" (disponível na rota `/docs`). O inventário da nossa API é intrínseco ao código, garantindo que as rotas e os schemas de dados esperados estejam sempre 100% atualizados e visíveis para auditoria.

## 5. API8:2023 - Lack of Protection from Automated Threats (Scraping/Bots)
* **Ameaça:** Bots automatizados bombardeando o endpoint `/predict` para fazer engenharia reversa das predições do nosso modelo LSTM e criar um modelo concorrente não autorizado.
* **Mitigação:** Como mitigação em nível de infraestrutura, a API foi empacotada de forma "Stateless", pronta para ser deployada atrás de um API Gateway ou WAF (Web Application Firewall) no ambiente de cloud da empresa, onde regras de *Rate Limiting* (Limite de requisições por IP) e bloqueio de tráfego de botnets devem ser aplicadas.
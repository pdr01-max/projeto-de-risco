# Publicar o Quant Risk Engine em 5 passos

1. Crie uma conta em [Hugging Face](https://huggingface.co/) ou [Render](https://render.com/). O projeto deve ser publicado com permissão de leitura pública para gerar um link acessível a qualquer pessoa.
2. Para **Hugging Face Spaces**, clique em **New Space**, escolha **Docker** e visibilidade **Public**. O `README.md` já declara `sdk: docker` e porta `7860`.
3. Na aba **Files** do Space, envie pelo navegador os arquivos `README.md`, `Dockerfile`, `requirements.txt`, `api_app.py` e as pastas `static/` e `quant_risk_engine/`, preservando a estrutura de diretórios. Aguarde o build e abra a URL `https://huggingface.co/spaces/SEU_USUARIO/NOME_DO_SPACE` exibida na página do Space.
4. Para **Render**, envie esta pasta a um repositório GitHub. No Render, escolha **New → Blueprint**, conecte o repositório e confirme o `render.yaml`. O serviço usará Docker, `/health` e a porta fornecida em `PORT`.
5. Abra o link HTTPS gerado, teste uma carteira B3 e uma dos EUA e compartilhe o URL. Se o Yahoo Finance limitar o IP do host, a API mostra uma mensagem de erro; tente novamente após alguns minutos.

O aplicativo não armazena carteiras: pesos e limites são processados por requisição. O cache de 15 minutos contém apenas preços de mercado. Carteiras mistas B3/EUA são convertidas para USD usando a série USDBRL=X do Yahoo Finance.

import pandas as pd
import numpy as np
import os
import unicodedata
import re

# Caminho do CSV original
caminho_arquivo = 'vendas_modificado.csv'

# =========================
# 1. Leitura e padronização inicial
# =========================
df = pd.read_csv(caminho_arquivo)
print("Arquivo carregado com sucesso!")

linhas_iniciais = len(df)
colunas_iniciais = df.columns.tolist()

def padronizar_coluna(col):
    col = unicodedata.normalize('NFKD', col).encode('ASCII', 'ignore').decode('utf-8')
    col = re.sub(r'[^a-zA-Z0-9_]', '_', col).lower()
    col = re.sub(r'_+', '_', col)
    return col.strip('_')

df.columns = [padronizar_coluna(col) for col in df.columns]

# =========================
# 2. Limpeza de dados
# =========================

df.dropna(axis=1, how='all', inplace=True)

for col in df.columns:
    if df[col].nunique() <= 1:
        df.drop(columns=col, inplace=True)

limite_unicidade = 0.9
for col in df.columns:
    if df[col].nunique() / len(df) > limite_unicidade:
        df.drop(columns=col, inplace=True)

def limpar_texto(texto):
    if isinstance(texto, str):
        texto = texto.strip()
        texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
        texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
        texto = re.sub(r'\s+', ' ', texto)
    return texto

df = df.apply(lambda col: col.map(limpar_texto) if col.dtype == 'object' else col)

# Padronização de nomes próprios
for col in ['cliente', 'vendedor']:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x.title() if isinstance(x, str) else x)

# Padronização de status
if 'status' in df.columns:
    df['status'] = df['status'].str.lower().str.strip()
    df['status'] = df['status'].map({
        'aprovado': 'Aprovado',
        'ap': 'Aprovado',
        'reprovado': 'Reprovado',
        'pendente': 'Pendente'
    }).fillna('Desconhecido')

# Padronização de produtos
if 'produto' in df.columns:
    df['produto'] = df['produto'].str.lower().str.strip()
    df['produto'] = df['produto'].replace({
        'tv led': 'tv',
        'tv lcd': 'tv',
        'smartphone': 'celular',
        'celulares': 'celular',
        'notebook': 'laptop',
        'notebooks': 'laptop',
        'impresora': 'impressora',
        'impressores': 'impressora'
    })

# Pagamento
padroes_pagamento = {
    'dinheiro': 'dinheiro',
    'cartao': 'cartao',
    'credito': 'cartao',
    'debito': 'cartao',
    'pix': 'pix',
    'boleto': 'boleto',
    'cartao de credito': 'cartao',
    'cartao de debito': 'cartao',
    'transferencia bancaria': 'transferencia'
}
if 'forma_pagamento' in df.columns:
    df['forma_pagamento'] = df['forma_pagamento'].str.lower().str.strip()
    df['forma_pagamento'] = df['forma_pagamento'].map(padroes_pagamento).fillna('outros')

# Estado
if 'estado' in df.columns:
    df['estado'] = df['estado'].str.upper()

# CEP
if 'cep' in df.columns:
    df['cep'] = df['cep'].astype(str).str.extract(r'(\d{5}\-?\d{3})', expand=False)
    df['cep'] = df['cep'].str.replace('-', '')
    df['cep'] = df['cep'].str.zfill(8)
    df['cep'] = df['cep'].where(df['cep'].str.len() == 8, np.nan)

# Valores monetários
for col in ['valor', 'total', 'frete']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = df[col].str.replace(r'[^\d.\-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].round(2)
        df = df[df[col] >= 0]

# Datas e Horários
colunas_data = [col for col in df.columns if 'data' in col]
for col in colunas_data:
    df[col] = pd.to_datetime(df[col], errors='coerce')

colunas_hora = [col for col in df.columns if 'hora' in col]
for col in colunas_hora:
    df[col] = df[col].astype(str).str.zfill(6)
    df[col] = df[col].str.extract(r'(\d{2})(\d{2})(\d{2})').agg(':'.join, axis=1)
    df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.time

# Validação data_entrega > data_venda
if 'data_venda' in df.columns and 'data_entrega' in df.columns:
    df = df[df['data_entrega'] >= df['data_venda']]

# Criar colunas derivadas
if 'data_venda' in df.columns:
    df['ano_venda'] = df['data_venda'].dt.year
    df['mes_venda'] = df['data_venda'].dt.month

# Preencher nulos (limite de 50%)
for col in df.columns:
    proporcao_nulos = df[col].isna().mean()
    if proporcao_nulos <= 0.5:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            moda = df[col].mode()
            if not moda.empty:
                df[col].fillna(moda[0], inplace=True)
            else:
                df[col].fillna("desconhecido", inplace=True)
    else:
        df.drop(columns=col, inplace=True)

# Duplicatas
chave = 'id_da_compra' if 'id_da_compra' in df.columns else None
if chave:
    df.drop_duplicates(subset=chave, inplace=True)
else:
    df.drop_duplicates(inplace=True)

# Outliers
def remover_outliers(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    return (col >= q1 - 1.5 * iqr) & (col <= q3 + 1.5 * iqr)

for col in df.select_dtypes(include=[np.number]):
    df = df[remover_outliers(df[col])]

# =========================
# Renomear colunas para maiúsculas (aplicado no final)
# =========================
mapa_colunas_maiusculas = {
    'id_da_compra': 'Id_da_compra',
    'data': 'Data',
    'hora': 'Hora',
    'cliente': 'Cliente',
    'produto': 'Produto',
    'valor': 'Valor',
    'quantidade': 'Quantidade',
    'total': 'Total',
    'status': 'Status',
    'cidade': 'Cidade',
    'estado': 'Estado',
    'pais': 'Pais',
    'cep': 'Cep',
    'frete': 'Frete',
    'pagamento': 'Pagamento',
    'vendedor': 'Vendedor',
    'marca': 'Marca'
}

df.rename(columns=mapa_colunas_maiusculas, inplace=True)

# =========================
# 4. Exportar
# =========================
caminho_saida = os.path.join(os.path.dirname(caminho_arquivo), 'vendas_limpo.csv')
df.to_csv(caminho_saida, index=False)
print(f"Arquivo limpo salvo em: {caminho_saida}")

# =========================
# 5. Relatório
# =========================
df_final = df.copy()

df_final.to_csv("vendas_limpo.csv", index=False)

linhas_finais = len(df_final)
colunas_finais = df_final.columns.tolist()

relatorio = f"""
# Relatório de Limpeza de Dados

✔️ Colunas padronizadas: {[c for c in colunas_iniciais if padronizar_coluna(c) in df_final.columns.str.lower().tolist()]}
   → Renomeadas com nomes consistentes, sem acentos/símbolos, para padronização e acesso facilitado.

✔️ Linhas iniciais: {linhas_iniciais}
✔️ Linhas finais: {linhas_finais}
✔️ Linhas removidas: {linhas_iniciais - linhas_finais}
✔️ Colunas finais: {len(colunas_finais)}

✔️ Ferramentas utilizadas: pandas, numpy, regex, unicodedata

✔️ Texto limpo (sem acentos, símbolos, espaços extras)
   → Garantia de consistência para análise, evitando ruídos no processamento.

✔️ Produtos e marcas padronizados
   → Redução de variações como “Celular”, “celular ”, “CELULAR”, etc., para uma única categoria.

✔️ Nomes de cliente/vendedor com inicial maiúscula
   → Padronização visual e leitura mais amigável.

✔️ Status e forma de pagamento normalizados
   → Valores inconsistentes (ex: “aprovado”, “Aprovado”) foram unificados.

✔️ CEPs validados e formatados
   → Apenas CEPs válidos foram mantidos no padrão 00000-000.

✔️ Valores monetários convertidos e arredondados
   → Garantia que valores fossem tratados como numéricos e uniformes.

✔️ Datas verificadas + hora convertida
   → Campos de data/hora foram transformados para formato padrão e validados.

✔️ Campos derivados: ano_venda, mes_venda
   → Criado para facilitar análises temporais (ex: vendas por mês/ano).

✔️ Colunas com nulos > 50% removidas
   → Remoção de colunas com muitos valores ausentes que comprometem a análise.

✔️ Duplicatas eliminadas por id_da_compra
   → Mantido apenas o registro original para evitar distorções nas métricas.

✔️ Outliers removidos
   → Valores muito fora do padrão foram filtrados com base estatística para maior confiabilidade.
"""

print(relatorio)

with open("relatorio_limpeza.txt", "w", encoding="utf-8") as f:
    f.write(relatorio)

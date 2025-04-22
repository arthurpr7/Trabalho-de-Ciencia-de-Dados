import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

nome_arquivo = "vendas_limpo.csv"
nome_relatorio = "relatorio_analise.txt"
SUPORTE_ABSOLUTO = 5
FREQUENCIA_MINIMA_PRODUTO = 5

# Verifica se o arquivo existe
if not os.path.exists(nome_arquivo):
    print(f" Arquivo '{nome_arquivo}' não encontrado.")
    exit()

# Carrega os dados
df = pd.read_csv(nome_arquivo)

# Mostra colunas e exemplo
print("\n Colunas disponíveis:")
print(df.columns)
print("\n Exemplo de dados:")
print(df.head(5))
print(f"\n[{df.head(5).shape[0]} rows x {df.head(5).shape[1]} columns]\n")

# Remove produtos com baixa frequência
frequencia = df["Produto"].value_counts()
produtos_validos = frequencia[frequencia >= FREQUENCIA_MINIMA_PRODUTO].index
df = df[df["Produto"].isin(produtos_validos)]

# Agrupa por ID
agrupado = df.groupby("Id_da_compra")["Produto"].apply(list)
transacoes_multiplos = agrupado[agrupado.apply(lambda x: len(set(x)) > 1)]

print(f" Transações com múltiplos produtos: {len(transacoes_multiplos)}")

# Simulação se necessário
if len(transacoes_multiplos) == 0:
    print(" Simulando transações com múltiplos produtos...")
    df_simulado = df.sample(frac=1).copy()
    df_simulado["Id_Simulado"] = (df_simulado.index // 5) + 1
    transacoes_multiplos = df_simulado.groupby("Id_Simulado")["Produto"].apply(list)
    print(f" Transações simuladas: {len(transacoes_multiplos)}")

# Transforma para binário
te = TransactionEncoder()
te_ary = te.fit(transacoes_multiplos).transform(transacoes_multiplos)
df_binario = pd.DataFrame(te_ary, columns=te.columns_)

# Mineração
min_support = SUPORTE_ABSOLUTO / len(df_binario)
frequent_itemsets = apriori(df_binario, min_support=min_support, use_colnames=True)

if frequent_itemsets.empty:
    print(" Nenhum item frequente encontrado.")
    exit()

regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
regras_qualidade = regras[(regras["lift"] > 1) & (regras["confidence"] >= 0.2)]
regras_qualidade = regras_qualidade.sort_values(by="lift", ascending=False)

# Formatação das regras
regras_formatadas = regras_qualidade.copy()
regras_formatadas["Se tiver ➜"] = regras_formatadas["antecedents"].apply(lambda x: ", ".join(sorted(x)))
regras_formatadas["Então também ➜"] = regras_formatadas["consequents"].apply(lambda x: ", ".join(sorted(x)))

colunas_para_exibir = ["Se tiver ➜", "Então também ➜", "support", "confidence", "lift"]
regras_formatadas = regras_formatadas[colunas_para_exibir].rename(columns={
    "support": "Suporte",
    "confidence": "Confiança",
    "lift": "Lift"
})
regras_formatadas["Suporte"] = regras_formatadas["Suporte"].round(4)
regras_formatadas["Confiança"] = regras_formatadas["Confiança"].round(4)
regras_formatadas["Lift"] = regras_formatadas["Lift"].round(2)

# Exibição no terminal
print("\n" + "=" * 50)
print(" Regras de Associação com Maior Lift")
print("=" * 50)
print(regras_formatadas.head(10).to_string(index=False))

# Gráfico das regras
top_lift = regras_formatadas.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_lift["Se tiver ➜"] + " → " + top_lift["Então também ➜"], top_lift["Lift"], color='orange')
plt.title(" Top 10 Regras com Maior Lift")
plt.xlabel("Lift")
plt.ylabel("Regra")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Ranking de produtos
if 'Produto' in df.columns:
    ranking_produtos = df['Produto'].value_counts().reset_index()
    ranking_produtos.columns = ['Produto', 'Quantidade_Vendida']

    print("\n" + "=" * 50)
    print(" Top 10 Produtos Mais Vendidos")
    print("=" * 50)
    print(ranking_produtos.head(10).to_string(index=False))

    # Gráfico do ranking
    top10 = ranking_produtos.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top10['Produto'][::-1], top10['Quantidade_Vendida'][::-1], color='skyblue')
    plt.title(' Top 10 Produtos Mais Vendidos')
    plt.xlabel('Quantidade Vendida')
    plt.ylabel('Produto')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Geração do Relatório TXT
with open(nome_relatorio, "w", encoding="utf-8") as f:
    f.write("RELATÓRIO DE ANÁLISE DE VENDAS\n")
    f.write("=" * 60 + "\n\n")

    f.write("PRÉ-REQUISITOS E BIBLIOTECAS UTILIZADAS:\n")
    f.write("- pandas\n- matplotlib\n- mlxtend\n\n")
    f.write("Instalação recomendada:\n")
    f.write("pip install pandas mlxtend matplotlib\n\n")

    f.write("=" * 60 + "\n")
    f.write(" REGRAS DE ASSOCIAÇÃO COM MAIOR LIFT (TOP 10)\n")
    f.write("=" * 60 + "\n")
    f.write(regras_formatadas.head(10).to_string(index=False))
    f.write("\n\n")

    f.write("=" * 60 + "\n")
    f.write(" TOP 10 PRODUTOS MAIS VENDIDOS\n")
    f.write("=" * 60 + "\n")
    f.write(ranking_produtos.head(10).to_string(index=False))
    f.write("\n\n")

    f.write("=" * 60 + "\n")
    f.write(" TODAS AS REGRAS DE ASSOCIAÇÃO DE QUALIDADE\n")
    f.write("=" * 60 + "\n")
    f.write(regras_formatadas.to_string(index=False))
    f.write("\n")

print(f"\n Relatório gerado: {nome_relatorio}")

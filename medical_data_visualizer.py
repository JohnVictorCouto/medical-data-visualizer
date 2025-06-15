import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importar os dados
df = pd.read_csv('medical_examination.csv')

# 2. Adicionar coluna overweight (IMC > 25 é 1, caso contrário 0)
# Peso / (altura em metros)^2
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3. Normalizar colesterol e gluc (0 = normal, 1 = acima do normal)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


def draw_cat_plot():
    # 4. Criar DataFrame para o cat plot usando pd.melt
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 5. Agrupar e contar os dados para cada combinação
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 6. Criar o gráfico categórico com seaborn
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 7. Salvar figura
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # 8. Limpar dados para o heatmap
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 9. Calcular matriz de correlação
    corr = df_heat.corr()

    # 10. Criar máscara para triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 11. Configurar figura
    fig, ax = plt.subplots(figsize=(12, 10))

    # 12. Plotar heatmap
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, square=True, linewidths=0.5, ax=ax, center=0)

    # 13. Salvar figura
    fig.savefig('heatmap.png')
    return fig

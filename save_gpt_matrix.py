import json

# ваш список
data = 

# сохраняем в файл
with open('testing_final_20201012/testing_data_hmt_20201012_final/gpt_matrix_dev/a30.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# потом читаем обратно
# with open('relation_graph.json', 'r', encoding='utf-8') as f:
#     loaded_data = json.load(f)
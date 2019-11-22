import pandas as pd

# xgb_prob = pd.read_csv('result/lgb_prob.csv')[['id','target']]
xgb_prob = pd.read_csv('result/NN_EntityEmbed_10fold-sub.csv')[['id','target']]
print(xgb_prob)

xgb_prob['key']=[i for i in range(len(xgb_prob))]
xgb_prob.sort_values(by='target', inplace=True,ascending=False)

targets = []
for i in range(len(xgb_prob)):
    # if i <= 292287:
    # if i <= 283651:
    if i <= 404392.8:
        targets.append(1)
    else:
        targets.append(0)
xgb_prob['label'] = targets

print(xgb_prob.head(10000))
print(xgb_prob['label'].value_counts())

xgb_prob.sort_values(by='key',inplace=True,ascending=True)

xgb_prob[['id','label']].to_csv('result/xgb_result_ranking.csv', index=False,header=['id','target'])


submit = xgb_prob[['id']]
submit['target'] = xgb_prob['target']

submit['target'] = submit['target'].rank()
submit['target'] = (submit['target'] >= submit.shape[0] * 0.8934642948637943).astype(int)
submit.to_csv("result/submit.csv", index=False)

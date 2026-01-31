from sklearn.model_selection import train_test_split

train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

print("Train:", len(train_pairs))
print("Val:", len(val_pairs))

import numpy as np
from hmmlearn import hmm
from faker import Faker
import pandas as pd

fake = Faker()

num_customers = 1000
transactions_per_customer = 20

customer_profiles = ['Saver', 'Spender', 'Consistent Payer', 'Erratic']
transaction_categories = ['Groceries', 'Entertainment', 'Utilities', 'Dining', 'Travel']
num_profiles = len(customer_profiles)


transition_matrix = np.array([
    [0.6, 0.2, 0.1, 0.1],  # Saver
    [0.1, 0.6, 0.2, 0.1],  # Spender
    [0.2, 0.1, 0.6, 0.1],  # Consistent Payer
    [0.3, 0.2, 0.2, 0.3]   # Erratic
])

emission_probabilities = np.array([
    [0.8, 0.2],  # Saver
    [0.3, 0.7],  # Spender
    [0.5, 0.5],  # Consistent Payer
    [0.5, 0.5]   # Erratic
])


model = hmm.MultinomialHMM(n_components=num_profiles,n_trials=len(transition_matrix[0])-1)
model.startprob_ = np.array([0.25, 0.25, 0.25, 0.25])  # Equal probability to start with any profile
model.transmat_ = transition_matrix
model.emissionprob_ = emission_probabilities

transactions = []

for _ in range(num_customers):
    customer_name = fake.name()
    account_name = fake.iban()
    
    # Generate the initial profile for the customer
    profile_states, _ = model.sample(transactions_per_customer)
    profiles = [customer_profiles[state] for state in profile_states.ravel()]
    
    for idx, profile in enumerate(profiles):
        profile_index = customer_profiles.index(profile)
        transaction_emission_probs = emission_probabilities[profile_index]
        

        transaction_type = np.random.choice(['withdrawal', 'deposit'], p=transaction_emission_probs)
        

        transaction_time = fake.date_time_this_year()
        transaction_amount = np.random.randint(100, 1000) if transaction_type == 'deposit' else np.random.randint(20, 500)


        transaction_city = fake.city()
        transaction_category = np.random.choice(transaction_categories)
        transaction_ref_id = f"TXN-{fake.unique.random_number(digits=8)}-{idx}"
        
        transaction = {
            'customer_name': customer_name,
            'account_name': account_name,
            'profile': profile,
            'transaction_type': transaction_type,
            'Amount(United States Dollar)': transaction_amount,
            'time_transaction': transaction_time,
            'transaction_city': transaction_city,
            'transaction_category': transaction_category,
            'transaction_ref_id': transaction_ref_id
        }
        
        transactions.append(transaction)

new_df = pd.DataFrame(transactions)
new_df.to_csv('new_transaction_data_profiles.csv', index=False)


profile_counts = new_df['profile'].value_counts()
print("Profile Distribution:\n", profile_counts)




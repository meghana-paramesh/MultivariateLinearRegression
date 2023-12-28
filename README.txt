step1: create a new virtual environment for python (we can also use conda default "base" environment). Example: conda activate base or python -m venv <virtual_environment_name>
step 1.1: If the virtual environment is created using python -m venv <virtual_environment_name>, please activate it using the command "source assignment2_env/bin/activate" (for linux based systems)
step2: install all the requirements from "requirement.txt". Command: pip install -r requirements.txt
step3: run the main file that is Main.py. Command: python Main.py
step4: we can validate the plots generate "training_error.png" and "output.txt"
step5: we can also validate if the loss is decreasing using training_error.png
step6: Validate the output on the console. I have also added the final values of thetas and predictions
Stopping condition used:  l2_norm_based
=======================================
Gradient Descent Method
=======================================
Values of theta using Gradient Descent Method
theta_0:  340412.65957446524
theta_1:  [110631.05011541]
theta_2:  [-6649.47410738]
price prediction for a 1650-square-foot house with 3 bedrooms using gradient descent:  [293081.46437046]
=======================================
Normal Equation Method
=======================================
Values of theta using Normal equation in the format [[theta_0][theta_1][theta_2]]
[[89597.9095428 ]
 [  139.21067402]
 [-8738.01911233]]
price prediction for a 1650-square-foot house with 3 bedrooms using normal equations:  [293081.46433489]
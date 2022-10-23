import os
import MLP
import gurobipy as gp

solved = 0
too_large = 0
invalid_input = 0
total = 0
for i in range(1, 19):
    directory = f"Instances/Instances_{i}"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        # checking if it is a file
        if os.path.isfile(f):
            f_notxt = f.strip(".txt")
            f_no_instance = f_notxt.replace(f"Instances/Instances_{i}/", "")
            print(f"solving: {f}")
            total += 1
            try:
                MLP.solve_instance(f, f"solutions/{f_no_instance}-solution.txt")
                solved += 1

            except (gp.GurobiError, ValueError) as e:
                if "too large" in str(e):
                    too_large += 1

                elif "base 10" in str(e) or "could not convert" in str(e):
                    invalid_input += 1

                continue

print(f"The MLP solved: {solved} out of {total} instances \
	({round(solved / total * 100, 3)}%)")

print(f"{too_large} instances were too large for the MLP \
	({round(too_large / total * 100, 3)}%)")

print(f"{invalid_input} instances had invalid characters in their input \
	({round(invalid_input / total * 100, 3)}%)")
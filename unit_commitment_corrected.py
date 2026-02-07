"""
Unit Commitment Optimization Model
Fixed version - removes incorrect (model) call in objective function
"""

import pandas as pd
import pyomo.environ as pyo
import json


def calculate_expected_values(ren):
    """Calculate expected values for renewable generation scenarios."""
    expected_value_solar_list = []
    expected_value_wind_list = []
    for n in range(0, 5):
        expected_value_wind = sum(ren.loc[n*4:n*4+3, 'generation wind'] * 
                                   ren.loc[n*4:n*4+3, 'probability wind'])
        expected_value_solar = sum(ren.loc[n*4:n*4+3, 'generation solar'] * 
                                    ren.loc[n*4:n*4+3, 'probability solar'])
        expected_value_solar_list.append(expected_value_solar)
        expected_value_wind_list.append(expected_value_wind)
    return expected_value_wind_list, expected_value_solar_list


def create_renewable_scenario(ren):
    """Create renewable generation scenario dataframe."""
    hours = [n for n in range(0, 24)]
    generation = [0] * 24
    generation_df = pd.DataFrame({'hour': hours, 'generation': generation})
    expected_values = calculate_expected_values(ren)
    types = ['wind', 'solar']
    
    for n in range(0, 2):
        generation_df.loc[0:5, 'generation_' + types[n]] = expected_values[n][0]
        generation_df.loc[6:10, 'generation_' + types[n]] = expected_values[n][1]
        generation_df.loc[11:14, 'generation_' + types[n]] = expected_values[n][2]
        generation_df.loc[15:19, 'generation_' + types[n]] = expected_values[n][3]
        generation_df.loc[20:23, 'generation_' + types[n]] = expected_values[n][4]
    
    return generation_df


def objective_function(model):
    """Define the objective function for the optimization model."""
    obj = 0
    for n in model.n:
        # Generator costs
        generator_cost_1 = model.a1[0] * model.p1[n] + model.a2[0] * model.p1[n]**2
        generator_cost_2 = model.a1[1] * model.p2[n] + model.a2[1] * model.p2[n]**2
        generator_cost_3 = model.a1[2] * model.p3[n] + model.a2[2] * model.p3[n]**2
        generator_cost_4 = model.a1[3] * model.p4[n] + model.a2[3] * model.p4[n]**2
        generator_costs = generator_cost_1 + generator_cost_2 + generator_cost_3 + generator_cost_4
        
        # Imbalance costs
        imbalance_cost = model.imbalance[n] * 1000
        
        # Min/max violation costs
        min_costs = (model.p1_pen_min[n] + model.p2_pen_min[n] + 
                     model.p3_pen_min[n] + model.p4_pen_min[n])
        max_costs = (model.p1_pen_max[n] + model.p2_pen_max[n] + 
                     model.p3_pen_max[n] + model.p4_pen_max[n])
        
        obj += generator_costs + imbalance_cost + min_costs + max_costs
    
    return obj


# Generator 1 minimum constraints
def constraint_1_min_fun_1(model, index):
    return model.p1_pen_min[index] >= 0


def constraint_1_min_fun_2(model, index):
    return model.p1_pen_min[index] >= (gen_params.loc[0, 'pmin'] - model.p1[index]) * model.u1[index]


def constraint_1_min_fun_3(model, index):
    return model.p1_pen_min[index] <= M * model.c1_min[index]


def constraint_1_min_fun_4(model, index):
    return model.p1_pen_min[index] <= gen_params.loc[0, 'pmin'] - model.p1[index] + M * (1 - model.c1_min[index])


# Generator 1 maximum constraints
def constraint_1_max_fun_1(model, index):
    return model.p1_pen_max[index] >= 0


def constraint_1_max_fun_2(model, index):
    return model.p1_pen_max[index] >= model.p1[index] - gen_params.loc[0, 'pmax']


def constraint_1_max_fun_3(model, index):
    return model.p1_pen_max[index] <= M * model.c1_max[index]


def constraint_1_max_fun_4(model, index):
    return model.p1_pen_max[index] <= model.p1[index] - gen_params.loc[0, 'pmax'] + M * (1 - model.c1_max[index])


# Generator 2 minimum constraints
def constraint_2_min_fun_1(model, index):
    return model.p2_pen_min[index] >= 0


def constraint_2_min_fun_2(model, index):
    return model.p2_pen_min[index] >= (gen_params.loc[1, 'pmin'] - model.p2[index]) * model.u2[index]


def constraint_2_min_fun_3(model, index):
    return model.p2_pen_min[index] <= M * model.c2_min[index]


def constraint_2_min_fun_4(model, index):
    return model.p2_pen_min[index] <= gen_params.loc[1, 'pmin'] - model.p2[index] + M * (1 - model.c2_min[index])


# Generator 2 maximum constraints
def constraint_2_max_fun_1(model, index):
    return model.p2_pen_max[index] >= 0


def constraint_2_max_fun_2(model, index):
    return model.p2_pen_max[index] >= model.p2[index] - gen_params.loc[1, 'pmax']


def constraint_2_max_fun_3(model, index):
    return model.p2_pen_max[index] <= M * model.c2_max[index]


def constraint_2_max_fun_4(model, index):
    return model.p2_pen_max[index] <= model.p2[index] - gen_params.loc[1, 'pmax'] + M * (1 - model.c2_max[index])


# Generator 3 minimum constraints
def constraint_3_min_fun_1(model, index):
    return model.p3_pen_min[index] >= 0


def constraint_3_min_fun_2(model, index):
    return model.p3_pen_min[index] >= (gen_params.loc[2, 'pmin'] - model.p3[index]) * model.u3[index]


def constraint_3_min_fun_3(model, index):
    return model.p3_pen_min[index] <= M * model.c3_min[index]


def constraint_3_min_fun_4(model, index):
    return model.p3_pen_min[index] <= gen_params.loc[2, 'pmin'] - model.p3[index] + M * (1 - model.c3_min[index])


# Generator 3 maximum constraints
def constraint_3_max_fun_1(model, index):
    return model.p3_pen_max[index] >= 0


def constraint_3_max_fun_2(model, index):
    return model.p3_pen_max[index] >= model.p3[index] - gen_params.loc[2, 'pmax']


def constraint_3_max_fun_3(model, index):
    return model.p3_pen_max[index] <= M * model.c3_max[index]


def constraint_3_max_fun_4(model, index):
    return model.p3_pen_max[index] <= model.p3[index] - gen_params.loc[2, 'pmax'] + M * (1 - model.c3_max[index])


# Generator 4 minimum constraints
def constraint_4_min_fun_1(model, index):
    return model.p4_pen_min[index] >= 0


def constraint_4_min_fun_2(model, index):
    return model.p4_pen_min[index] >= (gen_params.loc[3, 'pmin'] - model.p4[index]) * model.u4[index]


def constraint_4_min_fun_3(model, index):
    return model.p4_pen_min[index] <= M * model.c4_min[index]


def constraint_4_min_fun_4(model, index):
    return model.p4_pen_min[index] <= gen_params.loc[3, 'pmin'] - model.p4[index] + M * (1 - model.c4_min[index])


# Generator 4 maximum constraints
def constraint_4_max_fun_1(model, index):
    return model.p4_pen_max[index] >= 0


def constraint_4_max_fun_2(model, index):
    return model.p4_pen_max[index] >= model.p4[index] - gen_params.loc[3, 'pmax']


def constraint_4_max_fun_3(model, index):
    return model.p4_pen_max[index] <= M * model.c4_max[index]


def constraint_4_max_fun_4(model, index):
    return model.p4_pen_max[index] <= model.p4[index] - gen_params.loc[3, 'pmax'] + M * (1 - model.c4_max[index])


# Constraints for u = 0 when power = 0
def constraint_u_rule_1(model, index):
    return model.p1[index] <= M * model.u1[index]


def constraint_u_rule_2(model, index):
    return model.p2[index] <= M * model.u2[index]


def constraint_u_rule_3(model, index):
    return model.p3[index] <= M * model.u3[index]


def constraint_u_rule_4(model, index):
    return model.p4[index] <= M * model.u4[index]


# Constraints for wind and solar
def wind_constraint(model, index):
    return model.wind_prod[index] == model.expected_wind[index] * model.uwind[index]


def solar_constraint(model, index):
    return model.solar_prod[index] == model.expected_solar[index] * model.usolar[index]


# Imbalance constraints
def constraint_imbalance_1(model, index):
    imbalance = (model.p1[index] + model.p2[index] + model.p3[index] + model.p4[index] - 
                 model.load[index] + model.wind_prod[index] + model.solar_prod[index])
    return imbalance == model.pos_imbalance[index] - model.neg_imbalance[index]


def constraint_imbalance_2(model, index):
    return model.imbalance[index] == model.pos_imbalance[index] + model.neg_imbalance[index]


def constraint_imbalance_3(model, index):
    return model.pos_imbalance[index] <= M * model.imbalance_c[index]


def constraint_imbalance_4(model, index):
    return model.neg_imbalance[index] <= M * (1 - model.imbalance_c[index])


def main():
    """Main function to run the unit commitment optimization."""
    global gen_params, M
    
    # Load data
    print("Loading data...")
    load = pd.read_csv('DataC_hourlyload.csv', sep=';')
    renewables = pd.read_csv('DataC_nondispgeneration.csv', sep=',')
    gen_params = pd.read_csv('DataC_generatorparams.csv', sep=';')
    
    # Process load data
    load['hour'] = load.index
    load = load.drop(['Start', 'End'], axis=1)
    load = load[['Date', 'hour', 'Total (grid load) [MWh] Calculated resolutions']]
    load.columns = ['Date', 'hour', 'load']
    
    # Create renewable scenario and merge with load
    renewable_expected = create_renewable_scenario(renewables)
    load = pd.merge(load, renewable_expected)
    
    # Big M constant
    M = 100000000
    
    # Create optimization model
    print("Building optimization model...")
    model = pyo.ConcreteModel()
    
    # Model parameters
    model.a1 = gen_params["a1"]
    model.a2 = gen_params["a2"]
    model.pmin = gen_params["pmin"]
    model.pmax = gen_params["pmax"]
    model.load = load['load']
    model.expected_wind = load['generation_wind']
    model.expected_solar = load['generation_solar']
    
    # Sets
    model.n = pyo.Set(initialize=range(0, 24))
    
    # Variables
    model.p1 = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.p2 = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.p3 = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.p4 = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.u1 = pyo.Var(model.n, domain=pyo.Binary)
    model.u2 = pyo.Var(model.n, domain=pyo.Binary)
    model.u3 = pyo.Var(model.n, domain=pyo.Binary)
    model.u4 = pyo.Var(model.n, domain=pyo.Binary)
    model.p1_pen_max = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.p2_pen_max = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.p3_pen_max = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.p1_pen_min = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.p2_pen_min = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.p3_pen_min = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.p4_pen_max = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.p4_pen_min = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.c1_min = pyo.Var(model.n, domain=pyo.Binary)
    model.c1_max = pyo.Var(model.n, domain=pyo.Binary)
    model.c2_min = pyo.Var(model.n, domain=pyo.Binary)
    model.c2_max = pyo.Var(model.n, domain=pyo.Binary)
    model.c3_min = pyo.Var(model.n, domain=pyo.Binary)
    model.c3_max = pyo.Var(model.n, domain=pyo.Binary)
    model.c4_min = pyo.Var(model.n, domain=pyo.Binary)
    model.c4_max = pyo.Var(model.n, domain=pyo.Binary)
    model.imbalance = pyo.Var(model.n, domain=pyo.Reals)
    model.uwind = pyo.Var(model.n, domain=pyo.Binary)
    model.usolar = pyo.Var(model.n, domain=pyo.Binary)
    model.wind_prod = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.solar_prod = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.imbalance_c = pyo.Var(model.n, domain=pyo.Binary)
    model.pos_imbalance = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    model.neg_imbalance = pyo.Var(model.n, domain=pyo.NonNegativeReals)
    
    # Constraints
    model.p1_min_1 = pyo.Constraint(model.n, rule=constraint_1_min_fun_1)
    model.p1_min_2 = pyo.Constraint(model.n, rule=constraint_1_min_fun_2)
    model.p1_min_3 = pyo.Constraint(model.n, rule=constraint_1_min_fun_3)
    model.p1_min_4 = pyo.Constraint(model.n, rule=constraint_1_min_fun_4)
    
    model.p2_min_1 = pyo.Constraint(model.n, rule=constraint_2_min_fun_1)
    model.p2_min_2 = pyo.Constraint(model.n, rule=constraint_2_min_fun_2)
    model.p2_min_3 = pyo.Constraint(model.n, rule=constraint_2_min_fun_3)
    model.p2_min_4 = pyo.Constraint(model.n, rule=constraint_2_min_fun_4)
    
    model.p3_min_1 = pyo.Constraint(model.n, rule=constraint_3_min_fun_1)
    model.p3_min_2 = pyo.Constraint(model.n, rule=constraint_3_min_fun_2)
    model.p3_min_3 = pyo.Constraint(model.n, rule=constraint_3_min_fun_3)
    model.p3_min_4 = pyo.Constraint(model.n, rule=constraint_3_min_fun_4)
    
    model.p4_min_1 = pyo.Constraint(model.n, rule=constraint_4_min_fun_1)
    model.p4_min_2 = pyo.Constraint(model.n, rule=constraint_4_min_fun_2)
    model.p4_min_3 = pyo.Constraint(model.n, rule=constraint_4_min_fun_3)
    model.p4_min_4 = pyo.Constraint(model.n, rule=constraint_4_min_fun_4)
    
    model.p1_max_1 = pyo.Constraint(model.n, rule=constraint_1_max_fun_1)
    model.p1_max_2 = pyo.Constraint(model.n, rule=constraint_1_max_fun_2)
    model.p1_max_3 = pyo.Constraint(model.n, rule=constraint_1_max_fun_3)
    model.p1_max_4 = pyo.Constraint(model.n, rule=constraint_1_max_fun_4)
    
    model.p2_max_1 = pyo.Constraint(model.n, rule=constraint_2_max_fun_1)
    model.p2_max_2 = pyo.Constraint(model.n, rule=constraint_2_max_fun_2)
    model.p2_max_3 = pyo.Constraint(model.n, rule=constraint_2_max_fun_3)
    model.p2_max_4 = pyo.Constraint(model.n, rule=constraint_2_max_fun_4)
    
    model.p3_max_1 = pyo.Constraint(model.n, rule=constraint_3_max_fun_1)
    model.p3_max_2 = pyo.Constraint(model.n, rule=constraint_3_max_fun_2)
    model.p3_max_3 = pyo.Constraint(model.n, rule=constraint_3_max_fun_3)
    model.p3_max_4 = pyo.Constraint(model.n, rule=constraint_3_max_fun_4)
    
    model.p4_max_1 = pyo.Constraint(model.n, rule=constraint_4_max_fun_1)
    model.p4_max_2 = pyo.Constraint(model.n, rule=constraint_4_max_fun_2)
    model.p4_max_3 = pyo.Constraint(model.n, rule=constraint_4_max_fun_3)
    model.p4_max_4 = pyo.Constraint(model.n, rule=constraint_4_max_fun_4)
    
    model.u1_rule = pyo.Constraint(model.n, rule=constraint_u_rule_1)
    model.u2_rule = pyo.Constraint(model.n, rule=constraint_u_rule_2)
    model.u3_rule = pyo.Constraint(model.n, rule=constraint_u_rule_3)
    model.u4_rule = pyo.Constraint(model.n, rule=constraint_u_rule_4)
    
    model.imbalance_1 = pyo.Constraint(model.n, rule=constraint_imbalance_1)
    model.imbalance_2 = pyo.Constraint(model.n, rule=constraint_imbalance_2)
    model.imbalance_3 = pyo.Constraint(model.n, rule=constraint_imbalance_3)
    model.imbalance_4 = pyo.Constraint(model.n, rule=constraint_imbalance_4)
    
    model.wind_cons = pyo.Constraint(model.n, rule=wind_constraint)
    model.solar_cons = pyo.Constraint(model.n, rule=solar_constraint)
    
    # Objective function (FIXED: removed incorrect (model) call)
    model.obj = pyo.Objective(rule=objective_function, sense=pyo.minimize)
    
    # Solve
    print("Solving optimization problem...")
    optim = pyo.SolverFactory('gurobi')
    result = optim.solve(model)
    
    # Display results
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    result.write()
    
    # Extract solution values
    u1_values = [float(val) if val else 0 for val in model.u1.get_values().values()]
    u2_values = [float(val) if val else 0 for val in model.u2.get_values().values()]
    u3_values = [float(val) if val else 0 for val in model.u3.get_values().values()]
    u4_values = [float(val) if val else 0 for val in model.u4.get_values().values()]
    p1_values = [float(val) if val else 0 for val in model.p1.get_values().values()]
    p2_values = [float(val) if val else 0 for val in model.p2.get_values().values()]
    p3_values = [float(val) if val else 0 for val in model.p3.get_values().values()]
    p4_values = [float(val) if val else 0 for val in model.p4.get_values().values()]
    uwind_values = [float(val) if val else 0 for val in model.uwind.get_values().values()]
    usolar_values = [float(val) if val else 0 for val in model.usolar.get_values().values()]
    
    print("\n" + "="*50)
    print("SOLUTION VALUES")
    print("="*50)
    print("u1:", u1_values)
    print("u2:", u2_values)
    print("u3:", u3_values)
    print("u4:", u4_values)
    print("p1:", p1_values)
    print("p2:", p2_values)
    print("p3:", p3_values)
    print("p4:", p4_values)
    print("uwind:", uwind_values)
    print("usolar:", usolar_values)
    
    # Save results to JSON
    data = {
        'name': 'Group_F',
        'task': 'C',
        'u1': [1 if val > 0 else 0 for val in u1_values],
        'u2': [1 if val > 0 else 0 for val in u2_values],
        'u3': [1 if val > 0 else 0 for val in u3_values],
        'u4': [1 if val > 0 else 0 for val in u4_values],
        'p1': p1_values,
        'p2': p2_values,
        'p3': p3_values,
        'p4': p4_values,
        'uwind': [1 if val > 0 else 0 for val in uwind_values],
        'usolar': [1 if val > 0 else 0 for val in usolar_values]
    }
    
    with open('results_F1.json', 'w') as json_file:
        json.dump(data, json_file, indent=2)
    
    print("\nResults saved to results_F1.json")
    print("="*50)


if __name__ == "__main__":
    main()

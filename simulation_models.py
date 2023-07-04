import attr
import pandas as pd
import math
from numpy_financial import npv
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

@attr.s(auto_attribs=True)
class PVBattBasic:
    # Read only fields

    # Simulation
    time_inc: float          #hrs

    # Profiles
    solar_profile: list
    load_profile: list
    grid_profile: list

    peak_demand: float  #kW ac

    # System parameters
    hybrid_system: bool    #T/F, False: off-grid
    cycle_battery: bool    #T/F, True: use battery to supply load if battery is high
    batt_ch_rating: float   #kW dc
    batt_dc_rating: float    #kW dc
    inv_rating: float        #kW ac, output
    grid_ch_rating: float    #kW dc, output
    soc_low: float           #%
    soc_start: float
    batt_capacity: float     #kWh

    # Grid connection
    grid_export_rating: float   #kW ac
    grid_import_rating: float  #kW ac

    # Efficiency
    e_pv: float
    e_inv: float
    e_gc: float
    e_bch: float
    e_bdc: float

    def __attrs_post_init__(self):

        # Create datetime array
        self.num_steps = int(8760/self.time_inc)
        start_dt = datetime.datetime(2023,1,1,0,0,0)
        self.dt_array = ([start_dt + datetime.timedelta(minutes=int(i*self.time_inc*60))
                     for i in range(self.num_steps)])
        
        # Initialise and empty dataframe for the outputs
        self.output_df_columns = [
            'solar', 'solar_for_load', 'solar_for_batt', 'solar_for_grid', 'solar_wasted', 'solar_input', 'solar_wasted_inv_limit',
            'load', 'load_from_solar', 'load_from_grid', 'load_from_batt',
            'grid', 'grid_for_batt', 'grid_for_load', 'grid_from_solar', 'grid_on',
            'loss_total', 'loss_pv', 'loss_gc', 'loss_inv', 'loss_batt',
            'batt_internal', 'batt', 'batt_soc', 'batt_soc_new'
        ]

        # Error limit for zero-sum checks
        self.max_error = 0.00000001

    def simulate(self):
        # Initialise output array
       
        batt_soc = self.soc_start
        output_arr = []
        
        for i in range(self.num_steps):
            # Get inputs for the time step
            solar_input = self.solar_profile[i]   #kW dc
            load = self.load_profile[i]           #kW ac
            grid_on = bool(self.grid_profile[i])
            
            # Simulate time step
            output_list = self.simulate_time_step(solar_input, load, batt_soc, grid_on)
            
            # Update battery SOC for next time step
            batt_soc = output_list[len(output_list)-1]

            # Save output data
            output_arr.append(output_list)
        
        self.output_df = pd.DataFrame(output_arr, columns=self.output_df_columns)
        self.output_df['datetime'] = self.dt_array

        return

    def simulate_time_step(self, solar_input, load, batt_soc, grid_on):
        # Battery status calculations (common)
        batt_high = batt_soc > self.soc_low

        batt_ch_max_to_full = (1-batt_soc) * self.batt_capacity / self.time_inc / self.e_bch
        batt_ch_max = min(batt_ch_max_to_full, self.batt_ch_rating)

        batt_dc_max_to_empty = -batt_soc * self.batt_capacity * self.e_bdc / self.time_inc
        batt_dc_max = max(batt_dc_max_to_empty, self.batt_dc_rating)

        # Solar 
        solar_limited = min(solar_input,self.inv_rating/self.e_inv/self.e_pv)
        solar_wasted_inv_limit = solar_input - solar_limited
        #print('to fix: solar_wasted_inv_limit needs to factor in battery charging')

        # System operating modes
        if self.hybrid_system:
            if grid_on:
                grid_export_max = self.grid_export_rating   #kW ac
                grid_import_max = self.grid_import_rating   #kW ac
                inverter_on = True
            else:   #power cut
                grid_export_max = 0 
                grid_import_max = 0
                inverter_on = True
        else:  #off-grid system
            if grid_on and not(batt_high):
                grid_export_max = 0 
                grid_import_max = self.grid_import_rating
                inverter_on = False
            elif grid_on and batt_high:
                grid_export_max = 0
                grid_import_max = self.grid_import_rating
                inverter_on = True*self.cycle_battery
            elif not(grid_on):
                grid_export_max = 0 
                grid_import_max = 0
                inverter_on = True
            else:
                raise Warning('grid_on or batt_high control logic error')

        # Solar balance, at solar_input, kW dc
        solar_for_load = min(load/self.e_inv/self.e_pv,
                             solar_limited) * inverter_on
        solar_for_batt = min(solar_limited - solar_for_load,
                             batt_ch_max/self.e_pv)
        solar_for_grid = min(solar_limited - solar_for_load - solar_for_batt,
                             grid_export_max)
        solar = solar_for_load + solar_for_batt + solar_for_grid
        solar_wasted = solar_limited - solar

        # Load balance, at load, kW ac
        load_from_solar = solar_for_load*self.e_pv*self.e_inv
        load_from_batt_initial = min(-batt_dc_max,
                                     load - load_from_solar)*batt_high*self.cycle_battery
        load_from_grid = min(grid_import_max, 
                             load - load_from_batt_initial - load_from_solar)
        load_from_batt = load - load_from_grid - load_from_solar


        # Battery balance, at battery input, kW dc
        # Charging
        batt_from_solar = solar_for_batt*self.e_pv
        if batt_high:
            batt_ch_target = min(batt_ch_max,batt_from_solar)
        else:
            batt_ch_target = batt_ch_max
        batt_from_grid = min(batt_ch_target - batt_from_solar, 
                             self.grid_ch_rating, 
                             (grid_import_max - load_from_grid)*self.e_gc)
        batt_ch = batt_from_solar + batt_from_grid

        # Discharging
        batt_dc = -load_from_batt/self.e_inv

        # Net
        if batt_ch > 0:
            batt = batt_ch
            batt_internal = batt_ch*self.e_bch
        else:
            batt = batt_dc
            batt_internal = batt_dc/self.e_bdc

        batt_energy_delta = batt_internal * self.time_inc
        batt_soc_new = batt_soc + batt_energy_delta / self.batt_capacity

        # Grid balance, at grid, ac
        grid_from_solar = -solar_for_grid * self.e_pv * self.e_inv #note: inverted (-)
        grid_for_load = load_from_grid
        grid_for_batt = batt_from_grid/self.e_gc
        grid = grid_for_batt + grid_for_load + grid_from_solar

        # Losses
        inverter_power_out = load_from_solar + load_from_batt - grid_from_solar
        inverter_power_in = inverter_power_out/self.e_inv

        loss_pv = solar*(1-self.e_pv)
        loss_gc = grid_for_batt*(1-self.e_gc)
        loss_inv = inverter_power_in*(1-self.e_inv)
        loss_batt = batt - batt_internal
        loss_total = loss_pv + loss_gc + loss_inv + loss_batt


        # Zero-sum checks
        # Load
        load_check = load - load_from_solar - load_from_batt - load_from_grid
        if abs(load_check) > self.max_error:
            print('solar check error:', load_check)
            print('solar:', solar, ' load:', load, ' soc:', batt_soc)
            print('load', load, ' load_from_solar:', load_from_solar,
                  ' load_from_batt:', load_from_batt, ' load_from_grid:', load_from_grid)
            raise Exception("load zero-check failed")

        # Solar
        solar_check = solar_input - solar_wasted_inv_limit - solar_for_load - solar_for_batt - solar_for_grid - solar_wasted
        if abs(solar_check) > self.max_error:
            print('solar check error:', solar_check)
            print('solar:', solar, ' load:', load, ' soc:', batt_soc)
            print('solar_input', solar_input, ' solar_wasted_inv_limit:', solar_wasted_inv_limit,
                  ' solar_for_load:', solar_for_load, ' solar_for_batt:', solar_for_batt,
                  ' solar_for_grid:', solar_for_grid, ' solar_wasted:', solar_wasted)
            raise Exception("solar zero-check failed")

        # Battery
        if batt < 0:
            batt_check = batt_dc - loss_batt - batt_internal
        else:
            batt_check = batt_ch - loss_batt - batt_internal
        if abs(batt_check) > self.max_error:
            print('batt check error:', batt_check)
            print('solar:', solar, ' load:', load, ' soc:', batt_soc)
            print('batt_ch', batt_ch, ' batt_dc:', batt_dc, 
                  ' loss_batt:', loss_batt, ' batt_internal:', batt_internal)
            raise Exception("batt zero-check failed")

        # Grid
        grid_check = grid - grid_for_batt - grid_for_load - grid_from_solar
        if abs(grid_check) > self.max_error:
            print('grid check error:', grid_check)
            print('solar:', solar, ' load:', load, ' soc:', batt_soc)
            print('gd:', grid, ' grid_for_batt:', grid_for_batt, ' grid_for_load:', grid_for_load, ' grid_from_solar:', grid_from_solar)
            raise Exception("grid zero-check failed")

        system_check = solar + grid - load - batt_internal - loss_total
        if abs(system_check) > self.max_error:
            print('system check error:', system_check)
            print('solar:', solar, ' load:', load, ' soc:', batt_soc)
            print('pv:', solar, ' gd:', grid, ' ld:', load, ' bt:', batt_internal, ' ls:', loss_total)
            raise Exception("system zero-check failed")
        
        # Make sure batt_soc_new is last in the list
        output_list = [solar, solar_for_load, solar_for_batt, solar_for_grid, solar_wasted, solar_input, solar_wasted_inv_limit,
                       load, load_from_solar, load_from_grid, load_from_batt,
                       grid, grid_for_batt, grid_for_load, grid_from_solar, grid_on,
                       loss_total, loss_pv, loss_gc, loss_inv, loss_batt,
                       batt_internal, batt, batt_soc, batt_soc_new]

        return output_list
    
    def graph_load(self, df, datetime_from, datetime_to):
        # Filter by date range
        df['graph_load_from_grid_base']=df['load_from_solar']+df['load_from_batt']
        df_graph = df[ (df['datetime']>=datetime_from) & (df['datetime']<datetime_to) ]

        # Load graph
        fig = go.Figure(
            data=[
                go.Bar(
                    name='PV to load',
                    x=df_graph['datetime'],
                    y=df_graph['load_from_solar'],
                    offsetgroup=0,
                    marker_color='#d75602',
                ),
                go.Bar(
                    name='Battery to load',
                    x=df_graph['datetime'],
                    y=df_graph['load_from_batt'],
                    offsetgroup=0,
                    base=df_graph['load_from_solar'],
                    marker_color='#9abca7',
                ),
                go.Bar(
                    name='Grid to load',
                    x=df_graph['datetime'],
                    y=df_graph['load_from_grid'],
                    offsetgroup=0,
                    base=df_graph['graph_load_from_grid_base'],
                    marker_color='#272635',
                ),
                go.Scatter(
                    name='Load',
                    x=df_graph['datetime'],
                    y=df_graph['load'],
                    marker_color='#fec22f',
                )
            ]
        )
        fig.update_yaxes(title_text="Energy (kWh)")
        fig.update_layout(legend=dict(orientation="h"))
        return fig

    def graph_solar(self, df, datetime_from, datetime_to):
        # Filter by date range
        df['graph_solar_for_grid_base'] = df['solar_for_load'] + df['solar_for_batt']
        df['graph_solar_wasted'] = df['solar_wasted'] + df['solar_wasted_inv_limit']
        df['graph_solar_wasted_base'] = df['graph_solar_for_grid_base'] + df['solar_for_grid']
        df_graph = df[ (df['datetime']>=datetime_from) & (df['datetime']<datetime_to) ]

        fig = go.Figure(
            data=[
                go.Bar(
                    name='PV to load',
                    x=df_graph['datetime'],
                    y=df_graph['solar_for_load'],
                    offsetgroup=0,
                    marker_color='#fec22f',
                ),
                go.Bar(
                    name='PV to battery',
                    x=df_graph['datetime'],
                    y=df_graph['solar_for_batt'],
                    offsetgroup=0,
                    base=df_graph['solar_for_load'],
                    marker_color='#9abca7',
                ),
                go.Bar(
                    name='PV to grid',
                    x=df_graph['datetime'],
                    y=df_graph['solar_for_grid'],
                    offsetgroup=0,
                    base=df_graph['graph_solar_for_grid_base'],
                    marker_color='#272635',
                ),
                go.Bar(
                    name='PV wasted',
                    x=df_graph['datetime'],
                    y=df_graph['graph_solar_wasted'],
                    offsetgroup=0,
                    base=df_graph['graph_solar_for_grid_base'],
                    marker_color='#f9cb9c'
                ),

                go.Scatter(
                    name='PV',
                    x=df_graph['datetime'],
                    y=df_graph['solar'],
                    marker_color='#d75602',
                )
            ]
        )
        fig.update_yaxes(title_text="Energy (kWh)")
        fig.update_layout(legend=dict(orientation="h"))
        return fig

    def graph_batt(self, df, datetime_from, datetime_to):
        # Filter by date range
        df['load_from_batt_inverse'] = -1*df['load_from_batt']
        df_graph = df[ (df['datetime']>=datetime_from) & (df['datetime']<datetime_to) ]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                name='Battery to load',
                x=df_graph['datetime'],
                y=df_graph['load_from_batt_inverse'],
                offsetgroup=0,
                marker_color='#fec22f',
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Bar(
                name='Battery from PV',
                x=df_graph['datetime'],
                y=df_graph['solar_for_batt'],
                offsetgroup=0,
                marker_color='#d75602',
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Bar(
                name='Battery from grid',
                x=df_graph['datetime'],
                y=df_graph['grid_for_batt'],
                offsetgroup=0,
                marker_color='#272635',
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                name='Battery power',
                x=df_graph['datetime'],
                y=df_graph['batt'],
                marker_color='#9abca7',
            ),
            secondary_y=False
        )        
        
        fig.add_trace(
            go.Scatter(
                name='State of charge',
                x=df_graph['datetime'],
                y=df_graph['batt_soc'],
                line = dict(color='red', width=1, dash='dash')
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            legend=dict(orientation="h"),
            yaxis=dict(
                title=dict(text="Energy (kWh)"),
                side="left",
                range=[-10, 10],
                dtick=2
            ),
            yaxis2=dict(
                title=dict(text="State of charge (%)"),
                side="right",
                range=[0, 1],
                dtick=0.1,
                overlaying="y",
                tickmode="sync",
            ),
        )
        return fig
    
    def graph_grid(self, df, datetime_from, datetime_to):
        df['graph_solar_for_grid_base']=df['solar_for_load']+df['solar_for_batt']
        df_graph = df[ (df['datetime']>=datetime_from) & (df['datetime']<datetime_to) ]

        fig = go.Figure(
            data=[
                go.Bar(
                    name='Grid to load',
                    x=df_graph['datetime'],
                    y=df_graph['grid_for_load'],
                    offsetgroup=0,
                    marker_color='#fec22f',
                ),
                go.Bar(
                    name='Grid to battery',
                    x=df_graph['datetime'],
                    y=df_graph['grid_for_batt'],
                    offsetgroup=0,
                    base=df_graph['grid_for_load'],
                    marker_color='#9abca7',
                ),
                go.Bar(
                    name='Grid from PV (export)',
                    x=df_graph['datetime'],
                    y=df_graph['grid_from_solar'],
                    offsetgroup=0,
                    marker_color='#d75602',
                ),
                go.Scatter(
                    name='Grid',
                    x=df_graph['datetime'],
                    y=df_graph['grid'],
                    marker_color='#272635',
                )
            ]
        )
        fig.update_yaxes(title_text="Energy (kWh)")
        fig.update_layout(legend=dict(orientation="h"))
        return fig
  
    def calc_monthly_grid_consumption(self):

        # Split grid column into +ve and -ve
        data = [None] * 12
        vals = self.output_df.grid.values
        zeros = np.full(len(self.output_df), 0)
        df2 = self.output_df[['datetime']].copy()
        df2['import'] = np.where(vals <0, zeros, vals)
        df2['export'] = np.where(vals >=0, zeros, vals)
        
        data = df2.groupby(pd.Grouper(key='datetime', freq='M')).agg(
            {
                'import': ['sum', 'min', 'max'],
                'export': ['sum', 'min', 'max']
            }
        )
        return data

    def calc_day_backup(self, day):
        # Incrementally increase hours and check backup
        hrs_backup_min = 4
        hrs_backup_max = 18

        for i in range(hrs_backup_max - hrs_backup_min):
            hrs_backup = hrs_backup_min + i

            hrs_start = 12 - math.floor(hrs_backup/2)
            hrs_end = 12 + math.ceil(hrs_backup/2)

            datetime_from = day + datetime.timedelta(hours = hrs_start)
            datetime_to = day + datetime.timedelta(hours = hrs_end)
            df = self.output_df[ (self.output_df['datetime']>=datetime_from) & (self.output_df['datetime']<datetime_to) ]

            load = hrs_backup * self.peak_demand
            solar = df['solar'].sum() + df['solar_wasted'].sum()
            batt = self.batt_capacity
            spare_capacity = solar + batt - load

            if spare_capacity < 0:
                # insufficient backup, roll back by 1 and exit
                hrs_backup -= 1
                break
        
        return hrs_backup








@attr.s(auto_attribs=True)
class CostBasic:
    # Read only parameters

    # Financial parameters
    project_years: int   #Total number of years for the project
    discount_rate: float #%

    # Cost inputs
    cost_pv: float      #Currency/kWpk
    cost_batt: float    #Currency/kWh
    cost_inv: float     #Currency/kWpk

    # Markup inputs
    markup_bos: float       #% markup on CAPEX above
    markup_install: float   #% markup on CAPEX above, inc. other markup above
    markup_transport: float #% markup on CAPEX above, inc. other markup above
    markup_margin: float    #% markup on CAPEX above, inc. other markup above

    # System details
    system_specs: dict                #dict{'pv_kW': 10, 'batt_kWh':100, ...}
    monthly_grid_consumption: object  #Pandas object with import/export sum/max/min

    # OPEX inputs
    opex_percentage: float #% of total CAPEX
    opex_inf: float  #% annual increase

    # Grid cost inputs
    net_metering: bool        #True to enable net_metering, 
                              #False assumes a PPA for all energy exported, and bill for all imported
    demand_import_slabs: list #% list [['slab_kW','slab_cost'],['slab_kW','slab_cost']]
    energy_import_slabs: list #% list [['slab_kWh','slab_cost'],['slab_kWh','slab_cost']]
    demand_export_slabs: list #% list [['slab_kW','slab_cost'],['slab_kW','slab_cost']]
    energy_export_slabs: list #% list [['slab_kWh','slab_cost'],['slab_kWh','slab_cost']]

    # REPEX inputs
    repex_inputs: list #list[['pv', {rep_yr: 25, rep_inf: 1.2}], ['batt', {rep_yr: 25, rep_inf: 1.2}],...]


    def calc_capex(self):
        # Return breakdown of costs in dictionary
        pv_cost = self.system_specs['pv_kW'] * self.cost_pv
        batt_cost = self.system_specs['batt_kWh'] * self.cost_batt
        inv_cost = self.system_specs['inv_kW'] * self.cost_inv
        self.capex = {
            'pv': pv_cost,
            'batt': batt_cost,
            'inv': inv_cost
        }
        return
    
    def calc_opex(self):
        # Return annual OPEX in array of length project_years
        om_yr1 = sum(list(self.capex.values())) * self.opex_percentage
        self.opex = [None] * self.project_years
        for i in range(self.project_years):
            if i == 0:
                self.opex[i] = om_yr1
            else:
                self.opex[i] = self.opex[i-1] * (1 + self.opex_inf)
        return
    
    def calc_repex(self):
        # Return annual OPEX in array of length project_years
        self.repex = [None] * self.project_years
        for i in range(self.project_years):
            year = i + 1
            repex = 0
            for component in self.repex_inputs:
                if component[1] == year:
                    component_type = component[0]
                    repex += self.capex[component_type] * component[2]

            self.repex[i] = repex

        return

    def get_monthly_volume_format(self, data_type):

        monthly_volume = [None] * len(self.monthly_grid_consumption)

        if data_type == 'energy':
            monthly_import = self.monthly_grid_consumption['import']['sum'].values
            monthly_export = self.monthly_grid_consumption['export']['sum'].values
        elif data_type == 'demand':
            monthly_import = self.monthly_grid_consumption['import']['max'].values
            monthly_export = self.monthly_grid_consumption['export']['min'].values
        else:
            raise Exception('error: data_type is neither "energy" nor "demand"')

        for i in range(len(monthly_volume)):
            monthly_volume[i] = [monthly_import[i],monthly_export[i]]

        return monthly_volume

    def calc_grid_costs(self):
        # Return annual REPEX in array of length project_years

        self.grid_energy_import_costs = self.calc_cost_by_slab(
            monthly_volume = self.get_monthly_volume_format(data_type = 'energy'),
            slabs = self.energy_import_slabs,
            grid_import = True,
            net_metering = self.net_metering,
        )

        self.grid_energy_export_costs = self.calc_cost_by_slab(
            monthly_volume = self.get_monthly_volume_format(data_type = 'energy'),
            slabs = self.energy_export_slabs,
            grid_import = False,
            net_metering = self.net_metering,
        )

        self.grid_demand_import_costs = self.calc_cost_by_slab(
            monthly_volume = self.get_monthly_volume_format(data_type = 'demand'),
            slabs = self.demand_import_slabs,
            grid_import = True,
            net_metering = False,
        )

        self.grid_demand_export_costs = self.calc_cost_by_slab(
            monthly_volume = self.get_monthly_volume_format(data_type = 'demand'),
            slabs = self.demand_export_slabs,
            grid_import = False,
            net_metering = False,
        )

        self.annual_grid_cost = {
            'demand_import': sum(self.grid_demand_import_costs['monthly_cost']),
            'demand_export': sum(self.grid_demand_export_costs['monthly_cost']),
            'energy_import': sum(self.grid_energy_import_costs['monthly_cost']),
            'energy_export': sum(self.grid_energy_export_costs['monthly_cost']),
        }

        self.annual_grid_volumes = {
            'demand_import': sum(self.grid_demand_import_costs['monthly_volume_max']),
            'demand_export': sum(self.grid_demand_export_costs['monthly_volume_min']),
            'energy_import': sum(self.grid_energy_import_costs['monthly_volume_total']),
            'energy_export': sum(self.grid_energy_export_costs['monthly_volume_total']),
        }

    def calc_cost_by_slab(self, monthly_volume, slabs, grid_import, net_metering):
        # Calculate grid costs for net energy import
        monthly_volume_by_slab = [None] * len(monthly_volume)
        monthly_cost_by_slab = [None] * len(monthly_volume) 
        monthly_volume_total = [None] * len(monthly_volume)
        monthly_volume_max = [None] * len(monthly_volume)
        monthly_volume_min = [None] * len(monthly_volume)
        monthly_cost = [None] * len(monthly_volume)

        for j in range(len(monthly_volume)):
            # Get data points for the month
            vol_import = monthly_volume[j][0]
            vol_export = monthly_volume[j][1]
            if vol_import < 0:
                raise Warning('vol_import < 0, should be +ve')
            if vol_export > 0:
                raise Warning('vol_export > 0, should be -ve')
            
            # Calculate volume to bill based on using net_metering calc or not
            if grid_import:     # Calculating import costs
                if net_metering:
                    volume = max(0,vol_import + vol_export)
                else:
                    volume = vol_import
            else:               # Calculating export costs (negative cost)
                if net_metering:
                    volume = min(0,vol_import + vol_export)
                    
                else:
                    volume = vol_export
            # Calculate the volume used and cost for each slab
            volume_by_slab = [None] * len(slabs)
            cost_by_slab = [None] * len(slabs)

            volume_abs = abs(volume)     

            if volume != 0:
                volume_sign = int(volume/volume_abs)   
            else:
                volume_sign = 1
            
             
            for i in range(len(slabs)):
                slab = slabs[i]
                if i == 0:
                    prev_slab = [0,0]
                else:
                    prev_slab = slabs[i-1]

                volume_above_prev_slab = volume_abs - prev_slab[0]
                slab_above_prev_slab = slab[0] - prev_slab[0]
                
                volume_by_slab[i] = max(0,min(volume_above_prev_slab, slab_above_prev_slab)) * volume_sign
                cost_by_slab[i] = volume_by_slab[i] * slab[1]

            # Save data in output array
            monthly_volume_by_slab[j] = volume_by_slab
            monthly_cost_by_slab[j] = cost_by_slab
            monthly_volume_total[j] = sum(volume_by_slab)
            monthly_volume_max[j] = max(volume_by_slab)
            monthly_volume_min[j] = min(volume_by_slab)
            monthly_cost[j] = sum(cost_by_slab)
        
        # Group data to pass to output
        data = {
            'monthly_volume_by_slab': monthly_volume_by_slab,
            'monthly_cost_by_slab': monthly_cost_by_slab,
            'monthly_volume_total': monthly_volume_total,
            'monthly_volume_max': monthly_volume_max,
            'monthly_volume_min': monthly_volume_min,
            'monthly_cost': monthly_cost,
            }

        return data

    def calc_npv(self):
        
        self.cashflow = [None] * (self.project_years+1)
        for i in range(len(self.cashflow)):
            if i == 0:
                self.cashflow[i] = sum(self.capex.values())
            else:
                self.cashflow[i] = self.opex[i-1] + self.repex[i-1] + sum(self.annual_grid_cost.values())
        
        self.npv = npv(self.discount_rate,self.cashflow)
        return 

@attr.s(auto_attribs=True)
class Simulations:
    # Read only parameters
    simulation_inputs: list # List of dictionaries with simulation inputs

    def run_simulations(self):
        self.simulation_outputs = [None] * len(self.simulation_inputs)
        
        for i in range(len(self.simulation_inputs)):
            self.simulation_outputs[i] = self.run_simulation(inputs = self.simulation_inputs[i])

    def run_simulation(self, inputs):
        # Inputs
        annual_demand_kWh = inputs['annual_demand_kWh']
        pv_to_load_ratio = inputs['pv_to_load_ratio']
        hybrid_system = inputs['hybrid_system']
        cycle_battery = inputs['cycle_battery']
        peak_demand_backup_hrs = inputs['peak_demand_backup_hrs']
        export_tariff = inputs['export_tariff'] 

        # Constants
        battery_min_c_rate = 5
        grid_export_rating = 100000   #kW ac
        grid_import_rating = 100000   #kW ac
        if cycle_battery:
            soc_low = 0.5          #%
        else:
            soc_low = 1
        soc_start = 0.5        #%
        # Efficiencies
        e_pv = 0.98
        e_inv = 0.95
        e_gc = 0.95
        e_bch = 0.95
        e_bdc = 0.95

        # Scale load profile
        load_profile_input = inputs['load_profile']
        load_scalar = annual_demand_kWh / sum(load_profile_input)
        load_profile = [i* load_scalar for i in load_profile_input]

        # Scale solar profile
        solar_target_generation = annual_demand_kWh * pv_to_load_ratio
        solar_profile_input = inputs['solar_profile']
        solar_scalar = solar_target_generation / sum(solar_profile_input)
        solar_profile = [i* solar_scalar for i in solar_profile_input]
        solar_capacity = sum(solar_profile)/1607.8
        print('to fix for non-mysore projects')

        # Extract grid profile
        grid_profile = inputs['grid_profile']

        # Size battery
        peak_demand = max(load_profile)
        batt_capacity = math.ceil(peak_demand * peak_demand_backup_hrs)
        print(batt_capacity)
        batt_ch_rating = math.ceil(batt_capacity/battery_min_c_rate)
        batt_dc_rating = -batt_ch_rating

        # Size inverter & grid charger
        grid_ch_rating = batt_ch_rating
        inv_rating = max(max(solar_profile),max(load_profile)) * 1.1
        
        les = PVBattBasic(
            time_inc = 1,           #hrs

            # Profiles
            solar_profile = solar_profile,
            load_profile = load_profile,
            grid_profile = grid_profile,

            peak_demand = peak_demand,

            # System specs
            hybrid_system = hybrid_system,   #T/F, False: off-grid
            cycle_battery = cycle_battery,
            batt_ch_rating = batt_ch_rating,     #kW dc
            batt_dc_rating = batt_dc_rating,   #kW dc
            inv_rating = inv_rating,       #kW ac, output
            grid_ch_rating = grid_ch_rating,    #kW dc, output
            soc_low = soc_low,          #%
            soc_start = soc_start,        #%
            batt_capacity = batt_capacity,    #kWh

            # Grid connection
            grid_export_rating = grid_export_rating,   #kW ac
            grid_import_rating = grid_import_rating,   #kW ac

            # Efficiency
            e_pv = e_pv,
            e_inv = e_inv,
            e_gc = e_gc,
            e_bch = e_bch,
            e_bdc = e_bdc,
        )

        # Run sumulation
        les.simulate()

        # Summarise data by month
        monthly_data = les.output_df.groupby(pd.Grouper(key='datetime', freq='M')).sum()
        monthly_data['datetime'] = monthly_data.index

        # Calculate minimum day time backup provided
        start_date = les.output_df['datetime'].min()
        end_date = les.output_df['datetime'].max()
        diff = end_date - start_date
        days_diff = diff.days + 1
        day_time_backup = [None] * days_diff

        for i in range(len(day_time_backup)):
            day = start_date + datetime.timedelta(days = i)
            day_time_backup[i] = les.calc_day_backup(day)
               
        les.day_time_backup_min = min(day_time_backup)

        ### Costs ###

        # Get monthly_grid_consumption for 
        monthly_grid_consumption = les.calc_monthly_grid_consumption()

        # Get system specs
        system_specs = {
            'pv_kW': solar_capacity,
            'batt_kWh': batt_capacity,
            'inv_kW': inv_rating,
        }
        
        if les.hybrid_system:
            cost_inv = 20000 #INR/kWpk
        else:
            cost_inv = 15000 #INR/kWpk

        cost = CostBasic(
            # Financial parameters
            project_years = 25,   #Total number of years for the project
            discount_rate = 0.07, #%

            # Cost inputs
            cost_pv = 27000,    #INR/kWpk
            cost_batt = 7000,   #INR/kWh
            cost_inv = cost_inv,   #INR/kWpk

            # Markup inputs
            markup_bos = 0.1,       #% markup on CAPEX above
            markup_install = 0.05,   #% markup on CAPEX above, inc. other markup above
            markup_transport = 0.05, #% markup on CAPEX above, inc. other markup above
            markup_margin = 0.15,    #% markup on CAPEX above, inc. other markup above

            # System details
            #dict{'pv_kW': 10, 'batt_kWh':100, ...}
            system_specs = system_specs,
            #list of length 12 with total demand/import {'import':[10000, 1000, ...,], 'export:[3000, ...]'}
            monthly_grid_consumption = monthly_grid_consumption,

            # OPEX inputs
            opex_percentage = 0.05, #% of total CAPEX
            opex_inf = 0.05, #% annual increase

            # Grid cost inputs
            net_metering = True,
            demand_import_slabs = [
                [50, 120],   #[peak kW demand, Rs/kW cost]
                [100, 150],
                [100000000, 175],
            ],
            demand_export_slabs = [
                [100000000, 0],   #[peak kW demand, Rs/kW cost]
            ],

            energy_import_slabs = [
                [200, 7.35], #[monthly kWh consumption, Rs/kWh cost]
                [400, 8],
                [100000000, 8.6],
            ],

            energy_export_slabs = [
                [100000000, export_tariff], #[monthly kWh consumption, Rs/kWh cost]
            ],

            # REPEX inputs
            #list['pv': {rep_yr: 25, rep_inf: 1.2}, 'batt': {rep_yr: 25, rep_inf: 1.2},...]
            repex_inputs = [
                # component, replacement year, replacement inflation
                ['pv', 26, 1.2],
                ['batt', 15, 1.2],
                ['inv', 15, 1.2],
            ],
        )

        cost.calc_capex()
        cost.calc_opex()
        cost.calc_repex()
        cost.calc_grid_costs()
        cost.calc_npv()
        
        outputs = {
            'hourly_simulation': les,
            'monthly_data': monthly_data,
            'costs': cost,
        }

        return outputs

    def create_output_table(self, root_folder):
        # Outputs in self.simulation_outputs array
        # Each array item has a dictionary
        #   {
        #       'hourly_simulation': les,
        #       'monthly_data': monthly_data,
        #       'costs': cost,
        #   }
        # les is an Object, includes hourly data in output_df 
        # monthly_data is the output_df summed by month
        # cost is an Object

        # Setup columns and empty data list for the dataframe
        columns = [
            'pv_kW', 'batt_kWh', 'inv_rating',
            'cost_capex', 'cost_capex_pv', 'cost_capex_batt', 'cost_capex_inv',
            'cost_grid_total', 'cost_import_kWh', 'cost_export_kWh', 'cost_import_kW', 'cost_export_kW', 
            'cost_opex_yr1', 'cost_repex_total', 'cost_npv',
            'an_load', 'an_solar', 'an_solar_wasted', 'an_grid', 'an_losses',
            'an_batt_ch', 'an_batt_dc', 'batt_day_backup_hrs',
            'an_grid_import_kWh', 'an_grid_export_kWh', 'an_grid_import_kW', 'an_grid_export_kW',
            ]
        data = [None] * len(self.simulation_outputs)
        
        # Get data to output from each simulation
        for i in range(len(self.simulation_outputs)):
            # Get data from simulation outputs
            les = self.simulation_outputs[i]['hourly_simulation']
            cost = self.simulation_outputs[i]['costs']

            # Data we want
            # System specs
            pv_kW = cost.system_specs['pv_kW']
            batt_kWh = cost.system_specs['batt_kWh']
            inv_rating = cost.system_specs['inv_kW']

            # Costs
            cost_capex_pv = cost.capex['pv']
            cost_capex_batt = cost.capex['batt']
            cost_capex_inv = cost.capex['inv']
            cost_capex = cost_capex_pv + cost_capex_batt + cost_capex_inv

            cost_opex_yr1 = cost.opex[0]
            cost_repex_total = sum(cost.repex)
            cost_import_kWh = cost.annual_grid_cost['energy_import']
            cost_export_kWh = cost.annual_grid_cost['energy_export']
            cost_import_kW = cost.annual_grid_cost['demand_import']
            cost_export_kW = cost.annual_grid_cost['demand_export']
            cost_grid_total = sum(cost.annual_grid_cost.values())
            cost_npv = cost.npv

            # Annual energy
            an_load = sum(les.output_df['load'])
            an_solar = sum(les.output_df['solar'])
            an_solar_wasted = sum(les.output_df['solar_wasted'])
            an_grid = sum(les.output_df['grid'])
            an_losses = sum(les.output_df['loss_total'])
            
            an_batt_ch = les.output_df[les.output_df.batt_internal.gt(0)]['batt_internal'].sum()
            an_batt_dc = les.output_df[les.output_df.batt_internal.lt(0)]['batt_internal'].sum()
            
            an_grid_import_kWh = cost.annual_grid_volumes['energy_import']
            an_grid_export_kWh = cost.annual_grid_volumes['energy_export']
            an_grid_import_kW = cost.annual_grid_volumes['demand_import']
            an_grid_export_kW = cost.annual_grid_volumes['demand_export']
            
            batt_day_backup_hrs = les.day_time_backup_min

            # Make sure this matches "columns" defined above
            data[i] = [
                pv_kW, batt_kWh, inv_rating,
                cost_capex, cost_capex_pv, cost_capex_batt, cost_capex_inv,
                cost_grid_total, cost_import_kWh, cost_export_kWh, cost_import_kW, cost_export_kW,
                cost_opex_yr1, cost_repex_total, cost_npv,
                an_load, an_solar, an_solar_wasted, an_grid, an_losses,
                an_batt_ch, an_batt_dc, batt_day_backup_hrs,
                an_grid_import_kWh, an_grid_export_kWh, an_grid_import_kW, an_grid_export_kW,
            ]

        # Build the dataframe and output
        self.output_table = pd.DataFrame(data, columns=columns)
        
        self.output_table.to_csv(root_folder + '/output_table.csv')
    
    def create_graphs(self, root_folder, export_png, export_html):
        # Plot day profile
        self.graphs = [None] * len(self.simulation_outputs)       
        for i in range(len(self.simulation_outputs)):

            #print('graphs for simulation ' + str(i))
            simulation = self.simulation_outputs[i]

            datetime_from = datetime.datetime(2023,7,1,0,0,0)
            datetime_to = datetime.datetime(2023,7,2,0,0,0)
            les = simulation['hourly_simulation']

            output_path = root_folder +'/' + 'sim' + str(i) +'/'
            #print('graphs in folder sim' + str(i))

            if not os.path.exists(output_path):
                #print('making output_path')
                os.mkdir(output_path)

            #print('building figures for single day')
            fig1 = les.graph_load(les.output_df, datetime_from=datetime_from, datetime_to=datetime_to)
            fig2 = les.graph_solar(les.output_df, datetime_from=datetime_from, datetime_to=datetime_to)
            fig3 = les.graph_batt(les.output_df, datetime_from=datetime_from, datetime_to=datetime_to)
            fig4 = les.graph_grid(les.output_df, datetime_from=datetime_from, datetime_to=datetime_to)

            graphs_day = {
                'load': fig1,
                'solar': fig2,
                'batt': fig3,
                'grid': fig4,
            }

            if export_html:
                #print('exporting html')
                fig1.write_html(output_path + "day_load.html")
                fig2.write_html(output_path + "day_solar.html")
                fig3.write_html(output_path + "day_batt.html")
                fig4.write_html(output_path + "day_grid.html")
            if export_png:
                #print('exporting png')
                fig1.write_image(output_path + "day_load.png", engine="kaleido")
                fig2.write_image(output_path + "day_solar.png", engine="kaleido")
                fig3.write_image(output_path + "day_batt.png", engine="kaleido")
                fig4.write_image(output_path + "day_grid.png", engine="kaleido")
            
            # Plot monthly data

            datetime_from = datetime.datetime(2023,1,1,0,0,0)
            datetime_to = datetime.datetime(2024,1,1,0,0,0)

            monthly_data = simulation['monthly_data']

            #print('building figures for single year')
            fig1 = les.graph_load(monthly_data, datetime_from=datetime_from, datetime_to=datetime_to)
            fig2 = les.graph_solar(monthly_data, datetime_from=datetime_from, datetime_to=datetime_to)
            fig3 = les.graph_grid(monthly_data, datetime_from=datetime_from, datetime_to=datetime_to)
            fig4 = les.graph_batt(monthly_data, datetime_from=datetime_from, datetime_to=datetime_to)

            graphs_month = {
                'load': fig1,
                'solar': fig2,
                'grid': fig3,
                'batt': fig4,
            }
            
            if export_html:
                #print('exporting html')
                fig1.write_html(output_path + "month_load.html")
                fig2.write_html(output_path + "month_solar.html")
                fig3.write_html(output_path + "month_grid.html")
            if export_png:
                #print('exporting png')
                fig1.write_image(output_path + "month_load.png", engine="kaleido")
                fig2.write_image(output_path + "month_solar.png", engine="kaleido")
                fig3.write_image(output_path + "month_grid.png", engine="kaleido")

            self.graphs[i] = {
                'day': graphs_day,
                'month': graphs_month
            }


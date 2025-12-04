import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================

APP_NAME = "Polymer Aviation Flight Log"
DB_FILENAME = "polymer_aviation_log.json"

AIRCRAFT_OPTIONS = ["N74JX", "N618KF"]
APPROACH_TYPES = ['Vis', 'RNAV', 'ILS', 'LPV', 'Circle', 'Overlay']
SIGNATURE_OPTIONS = ["Todd Hotes", "Andrew Waller"]

# ==========================================
# DATA MODELS
# ==========================================

@dataclass
class CumulativeTotals:
    hobbs: float = 0.0
    landings: int = 0
    left_engine_cycles: int = 0
    right_engine_cycles: int = 0

@dataclass
class InitialTotals:
    hobbs: float = 0.0
    landings: int = 0
    left_engine_cycles: int = 0
    right_engine_cycles: int = 0

@dataclass
class FlightSegment:
    id: str
    date: str
    aircraft_id: str
    operated_by: str
    dep_id: str
    arr_id: str
    diversion_id: str
    pax: int
    
    # Times
    fms_takeoff: str
    fms_landing: str
    fms_total: str
    hobbs_out: float
    hobbs_in: float
    
    # Crew
    pic_initials: str
    sic_initials: str
    
    # Stats
    landings: int
    approach_type: str
    fuel_consumed: int
    left_engine_cycles: int
    right_engine_cycles: int
    
    # Calculated on runtime/save
    cumulative: CumulativeTotals = field(default_factory=CumulativeTotals)

    @property
    def segment_duration(self) -> float:
        return max(0.0, self.hobbs_in - self.hobbs_out)

@dataclass
class VorCheck:
    id: str
    flight_id: str
    date: str
    signature: str
    
    # VOR 1
    vor1_name: str
    vor1_freq: str
    vor1_error: int
    
    # VOR 2
    vor2_name: str
    vor2_freq: str
    vor2_error: int

# ==========================================
# LOGIC & MANAGER CLASS
# ==========================================

class FlightLogManager:
    def __init__(self):
        self.flights: List[FlightSegment] = []
        self.vor_checks: List[VorCheck] = []
        self.initial_totals = InitialTotals()
        self.load_data()

    def load_data(self):
        """Loads data from the JSON master file if it exists."""
        if not os.path.exists(DB_FILENAME):
            return

        try:
            with open(DB_FILENAME, 'r') as f:
                data = json.load(f)
                
            # Load Initials
            init_data = data.get('initialTotals', {})
            self.initial_totals = InitialTotals(
                hobbs=init_data.get('hobbs', 0.0),
                landings=init_data.get('landings', 0),
                left_engine_cycles=init_data.get('leftEngineCycles', 0),
                right_engine_cycles=init_data.get('rightEngineCycles', 0)
            )

            # Load Flights
            self.flights = []
            for f in data.get('flights', []):
                # Reconstruct object
                seg = FlightSegment(
                    id=f['id'], date=f['date'], aircraft_id=f['aircraftId'],
                    operated_by=f['operatedBy'], dep_id=f['depId'], arr_id=f['arrId'],
                    diversion_id=f.get('diversionId', ''), pax=f['pax'],
                    fms_takeoff=f['fmsTakeoff'], fms_landing=f['fmsLanding'],
                    fms_total=f['fmsTotal'], hobbs_out=f['hobbsOut'], hobbs_in=f['hobbsIn'],
                    pic_initials=f['picInitials'], sic_initials=f['sicInitials'],
                    landings=f['landings'], approach_type=f['approachType'],
                    fuel_consumed=f.get('fuelConsumed', 0),
                    left_engine_cycles=f['leftEngineCycles'], right_engine_cycles=f['rightEngineCycles']
                )
                self.flights.append(seg)

            # Load VOR Checks
            self.vor_checks = []
            for v in data.get('vorChecks', []):
                chk = VorCheck(
                    id=v['id'], flight_id=v['flightId'], date=v['date'], signature=v['signature'],
                    vor1_name=v['vor1Name'], vor1_freq=v['vor1Frequency'], vor1_error=v['vor1BearingError'],
                    vor2_name=v['vor2Name'], vor2_freq=v['vor2Frequency'], vor2_error=v['vor2BearingError']
                )
                self.vor_checks.append(chk)

            self.recalculate_logbook()

        except Exception as e:
            print(f"Error loading data: {e}")

    def save_data(self):
        """Saves current state to JSON master file."""
        self.recalculate_logbook()
        
        # Serialize Objects
        flights_json = []
        for f in self.flights:
            f_dict = asdict(f)
            # CamelCase conversion for consistency with the React App JSON structure
            f_export = {
                'id': f.id, 'date': f.date, 'aircraftId': f.aircraft_id,
                'operatedBy': f.operated_by, 'depId': f.dep_id, 'arrId': f.arr_id,
                'diversionId': f.diversion_id, 'pax': f.pax,
                'fmsTakeoff': f.fms_takeoff, 'fmsLanding': f.fms_landing, 'fmsTotal': f.fms_total,
                'hobbsOut': f.hobbs_out, 'hobbsIn': f.hobbs_in,
                'picInitials': f.pic_initials, 'sicInitials': f.sic_initials,
                'landings': f.landings, 'approachType': f.approach_type,
                'fuelConsumed': f.fuel_consumed,
                'leftEngineCycles': f.left_engine_cycles, 'rightEngineCycles': f.right_engine_cycles,
                'cumulative': asdict(f.cumulative)
            }
            flights_json.append(f_export)

        vor_json = []
        for v in self.vor_checks:
            v_export = {
                'id': v.id, 'flightId': v.flight_id, 'date': v.date, 'signature': v.signature,
                'vor1Name': v.vor1_name, 'vor1Frequency': v.vor1_freq, 'vor1BearingError': v.vor1_error,
                'vor2Name': v.vor2_name, 'vor2Frequency': v.vor2_freq, 'vor2BearingError': v.vor2_error
            }
            vor_json.append(v_export)

        initials_json = {
            'hobbs': self.initial_totals.hobbs,
            'landings': self.initial_totals.landings,
            'leftEngineCycles': self.initial_totals.left_engine_cycles,
            'rightEngineCycles': self.initial_totals.right_engine_cycles
        }

        master_data = {
            'flights': flights_json,
            'vorChecks': vor_json,
            'initialTotals': initials_json,
            'lastUpdated': datetime.now().isoformat()
        }

        with open(DB_FILENAME, 'w') as f:
            json.dump(master_data, f, indent=2)
        print("Data saved successfully.")

    def recalculate_logbook(self):
        """Core Logic: Sorts flights and calculates running totals starting from Initial Totals."""
        # Sort by date
        self.flights.sort(key=lambda x: x.date)

        current_hobbs = self.initial_totals.hobbs
        current_landings = self.initial_totals.landings
        current_l_cycles = self.initial_totals.left_engine_cycles
        current_r_cycles = self.initial_totals.right_engine_cycles

        for flight in self.flights:
            duration = flight.segment_duration
            
            current_hobbs += duration
            current_landings += flight.landings
            current_l_cycles += flight.left_engine_cycles
            current_r_cycles += flight.right_engine_cycles

            flight.cumulative = CumulativeTotals(
                hobbs=round(current_hobbs, 1),
                landings=current_landings,
                left_engine_cycles=current_l_cycles,
                right_engine_cycles=current_r_cycles
            )

    def add_flight(self, flight: FlightSegment):
        self.flights.append(flight)
        self.recalculate_logbook()
        self.save_data()

    def add_vor_check(self, check: VorCheck):
        self.vor_checks.append(check)
        self.save_data()

    def get_last_vor_check_date(self) -> Optional[str]:
        if not self.vor_checks:
            return None
        # Sort by date descending
        sorted_checks = sorted(self.vor_checks, key=lambda x: x.date, reverse=True)
        return sorted_checks[0].date

    @staticmethod
    def calculate_fms_total(start: str, end: str) -> str:
        if not start or not end: return "00:00"
        try:
            fmt = "%H:%M"
            t1 = datetime.strptime(start, fmt)
            t2 = datetime.strptime(end, fmt)
            if t2 < t1:
                t2 += timedelta(days=1)
            diff = t2 - t1
            minutes = int(diff.total_seconds() / 60)
            hours = minutes // 60
            mins = minutes % 60
            return f"{hours:02}:{mins:02}"
        except:
            return "00:00"

    @staticmethod
    def days_between(d1: str, d2: str) -> int:
        try:
            date1 = datetime.strptime(d1, "%Y-%m-%d")
            date2 = datetime.strptime(d2, "%Y-%m-%d")
            return abs((date1 - date2).days)
        except:
            return 0

# ==========================================
# CLI INTERFACE
# ==========================================

def get_input_choice(prompt: str, options: List[str]) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"{i+1}. {opt}")
    while True:
        try:
            choice = int(input("Select option: "))
            if 1 <= choice <= len(options):
                return options[choice-1]
        except ValueError:
            pass
        print("Invalid selection.")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    manager = FlightLogManager()

    while True:
        clear_screen()
        print(f"=== {APP_NAME} ===")
        print(f"Total Flights: {len(manager.flights)} | Total VOR Checks: {len(manager.vor_checks)}")
        print("\nMain Menu:")
        print("1. Log New Flight Segment")
        print("2. Record VOR Check")
        print("3. View Master Logbook (Table)")
        print("4. View Daily Summary")
        print("5. View VOR Check History")
        print("6. Configure Initial Totals (Transfer In)")
        print("7. Exit")

        choice = input("\nSelect an option: ")

        if choice == '1':
            log_flight(manager)
        elif choice == '2':
            log_vor(manager)
        elif choice == '3':
            view_logbook(manager)
        elif choice == '4':
            view_daily_summary(manager)
        elif choice == '5':
            view_vor_history(manager)
        elif choice == '6':
            configure_initials(manager)
        elif choice == '7':
            print("Goodbye.")
            break
        else:
            input("Invalid option. Press Enter to continue...")

def log_flight(manager: FlightLogManager):
    clear_screen()
    print("--- Log New Flight Segment ---")
    
    date = input(f"Date [{datetime.now().strftime('%Y-%m-%d')}]: ") or datetime.now().strftime('%Y-%m-%d')
    aircraft = get_input_choice("Select Aircraft:", AIRCRAFT_OPTIONS)
    operator = input("Operated By: ")
    dep = input("Departure (ICAO): ").upper()
    arr = input("Arrival (ICAO): ").upper()
    div = input("Diversion (Optional): ").upper()
    pax = int(input("PAX Count: ") or 0)
    
    print("\n--- Times ---")
    fms_out = input("FMS Takeoff (HH:MM): ")
    fms_in = input("FMS Landing (HH:MM): ")
    fms_total = manager.calculate_fms_total(fms_out, fms_in)
    
    hobbs_out = float(input("Hobbs Out: "))
    hobbs_in = float(input("Hobbs In: "))
    
    print("\n--- Crew & Stats ---")
    pic = input("PIC Initials: ").upper()
    sic = input("SIC Initials: ").upper()
    landings = int(input("Landings: ") or 1)
    appr = get_input_choice("Approach Type:", APPROACH_TYPES)
    
    l_cycles = int(input("Left Engine Cycles (This Leg): ") or 1)
    r_cycles = int(input("Right Engine Cycles (This Leg): ") or 1)
    
    print("\n--- Fuel ---")
    fuel = int(input("Fuel Consumed (lbs): ") or 0)

    import uuid
    new_id = str(uuid.uuid4())[:8]

    seg = FlightSegment(
        id=new_id, date=date, aircraft_id=aircraft, operated_by=operator,
        dep_id=dep, arr_id=arr, diversion_id=div, pax=pax,
        fms_takeoff=fms_out, fms_landing=fms_in, fms_total=fms_total,
        hobbs_out=hobbs_out, hobbs_in=hobbs_in,
        pic_initials=pic, sic_initials=sic,
        landings=landings, approach_type=appr, fuel_consumed=fuel,
        left_engine_cycles=l_cycles, right_engine_cycles=r_cycles
    )

    manager.add_flight(seg)
    print("Flight Logged Successfully.")
    
    # Optional VOR Check associated with flight
    if input("Record VOR Check for this flight? (y/n): ").lower() == 'y':
        log_vor(manager, flight_id=new_id, date=date)
    else:
        input("Press Enter to continue...")

def log_vor(manager: FlightLogManager, flight_id=None, date=None):
    clear_screen()
    print("--- Record Dual VOR Check ---")
    
    if not date:
        date = input(f"Date [{datetime.now().strftime('%Y-%m-%d')}]: ") or datetime.now().strftime('%Y-%m-%d')
    
    # Check 30 day logic
    last_date = manager.get_last_vor_check_date()
    if last_date:
        days = manager.days_between(date, last_date)
        print(f"Previous Check: {last_date} ({days} days ago)")
        if days > 30:
            print("!!! WARNING: VOR CHECK EXPIRED (>30 Days) !!!")
        else:
            print("Status: Current")
    
    print("\n--- VOR 1 ---")
    v1_name = input("VOR 1 Identifier: ").upper()
    v1_freq = input("VOR 1 Freq: ")
    v1_err = int(input("VOR 1 Bearing Error: "))
    
    print("\n--- VOR 2 ---")
    v2_name = input("VOR 2 Identifier: ").upper()
    v2_freq = input("VOR 2 Freq: ")
    v2_err = int(input("VOR 2 Bearing Error: "))
    
    sig = get_input_choice("Signature:", SIGNATURE_OPTIONS)

    import uuid
    chk = VorCheck(
        id=str(uuid.uuid4())[:8],
        flight_id=flight_id if flight_id else "",
        date=date,
        signature=sig,
        vor1_name=v1_name, vor1_freq=v1_freq, vor1_error=v1_err,
        vor2_name=v2_name, vor2_freq=v2_freq, vor2_error=v2_err
    )
    
    manager.add_vor_check(chk)
    print("VOR Check Recorded.")
    input("Press Enter to continue...")

def view_logbook(manager: FlightLogManager):
    clear_screen()
    flights = manager.flights[::-1] # Show newest first
    
    print(f"{'Date':<12} {'Aircraft':<10} {'Route':<15} {'Hobbs':<8} {'Tot Hobbs':<10} {'Lndg':<5} {'Tot Lndg':<10} {'Fuel':<6}")
    print("-" * 90)
    
    for f in flights:
        route = f"{f.dep_id}>{f.arr_id}"
        seg_time = f.segment_duration
        print(f"{f.date:<12} {f.aircraft_id:<10} {route:<15} {seg_time:<8.1f} {f.cumulative.hobbs:<10.1f} {f.landings:<5} {f.cumulative.landings:<10} {f.fuel_consumed:<6}")
    
    input("\nPress Enter to return...")

def view_daily_summary(manager: FlightLogManager):
    clear_screen()
    date_str = input(f"Enter Date to View (YYYY-MM-DD): ")
    
    day_flights = [f for f in manager.flights if f.date == date_str]
    
    if not day_flights:
        print("No flights found for this date.")
    else:
        total_time = sum(f.segment_duration for f in day_flights)
        total_lndg = sum(f.landings for f in day_flights)
        total_fuel = sum(f.fuel_consumed for f in day_flights)
        total_l_cyc = sum(f.left_engine_cycles for f in day_flights)
        total_r_cyc = sum(f.right_engine_cycles for f in day_flights)
        aircrafts = ", ".join(set(f.aircraft_id for f in day_flights))
        
        print(f"\n=== Daily Summary: {date_str} ===")
        print(f"Aircraft Flown: {aircrafts}")
        print(f"Total Flight Time: {total_time:.1f} hrs")
        print(f"Total Landings:    {total_lndg}")
        print(f"Total Fuel:        {total_fuel} lbs")
        print(f"Total Cycles (L/R):{total_l_cyc} / {total_r_cyc}")
        
    input("\nPress Enter to return...")

def view_vor_history(manager: FlightLogManager):
    clear_screen()
    checks = sorted(manager.vor_checks, key=lambda x: x.date, reverse=True)
    
    print(f"{'Date':<12} {'VOR 1':<15} {'Err 1':<6} {'VOR 2':<15} {'Err 2':<6} {'Signature':<15}")
    print("-" * 80)
    
    for c in checks:
        v1 = f"{c.vor1_name} ({c.vor1_freq})"
        v2 = f"{c.vor2_name} ({c.vor2_freq})"
        print(f"{c.date:<12} {v1:<15} {c.vor1_error:<6} {v2:<15} {c.vor2_error:<6} {c.signature:<15}")

    input("\nPress Enter to return...")

def configure_initials(manager: FlightLogManager):
    clear_screen()
    print("--- Transfer In / Initial Totals ---")
    print("Current Initials:")
    print(manager.initial_totals)
    print("\nEnter new starting values:")
    
    try:
        h = float(input("Total Hobbs: "))
        l = int(input("Total Landings: "))
        lc = int(input("Left Engine Cycles: "))
        rc = int(input("Right Engine Cycles: "))
        
        manager.initial_totals = InitialTotals(h, l, lc, rc)
        manager.save_data() # This triggers recalculation
        print("Totals updated and logbook recalculated.")
    except ValueError:
        print("Invalid input. No changes made.")
    
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()

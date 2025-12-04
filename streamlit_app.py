import streamlit as st
import pandas as pd
import json
import os
import uuid
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

st.set_page_config(
    page_title="Polymer Aviation Flight Log",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILENAME = "polymer_aviation_log.json"
AIRCRAFT_OPTIONS = ["N74JX", "N618KF"]
APPROACH_TYPES = ['Vis', 'RNAV', 'ILS', 'LPV', 'Circle', 'Overlay']
SIGNATURE_OPTIONS = ["Todd Hotes", "Andrew Waller"]

# ==========================================
# DATA MODELS (Equivalent to types.ts)
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
    fms_takeoff: str
    fms_landing: str
    fms_total: str
    hobbs_out: float
    hobbs_in: float
    pic_initials: str
    sic_initials: str
    landings: int
    approach_type: str
    fuel_consumed: int
    left_engine_cycles: int
    right_engine_cycles: int
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
    vor1_name: str
    vor1_freq: str
    vor1_error: int
    vor2_name: str
    vor2_freq: str
    vor2_error: int

# ==========================================
# LOGIC MANAGER (Equivalent to utils.ts)
# ==========================================

class FlightLogManager:
    def __init__(self):
        self.flights: List[FlightSegment] = []
        self.vor_checks: List[VorCheck] = []
        self.initial_totals = InitialTotals()
        self.load_data()

    def load_data(self):
        """Loads data from local JSON file."""
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
                # Rehydrate cumulative if it exists, otherwise recalc will handle it
                if 'cumulative' in f:
                    c = f['cumulative']
                    seg.cumulative = CumulativeTotals(c['hobbs'], c['landings'], c['leftEngineCycles'], c['rightEngineCycles'])
                self.flights.append(seg)

            # Load VOR Checks
            self.vor_checks = []
            for v in data.get('vorChecks', []):
                chk = VorCheck(
                    id=v['id'], flight_id=v.get('flightId', ''), date=v['date'], signature=v['signature'],
                    vor1_name=v['vor1Name'], vor1_freq=v['vor1Frequency'], vor1_error=v['vor1BearingError'],
                    vor2_name=v['vor2Name'], vor2_freq=v['vor2Frequency'], vor2_error=v['vor2BearingError']
                )
                self.vor_checks.append(chk)

            self.recalculate_logbook()

        except Exception as e:
            st.error(f"Error loading data: {e}")

    def save_data(self):
        """Saves current state to local JSON file."""
        self.recalculate_logbook()
        
        flights_json = []
        for f in self.flights:
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

        master_data = {
            'flights': flights_json,
            'vorChecks': vor_json,
            'initialTotals': asdict(self.initial_totals),
            'lastUpdated': datetime.now().isoformat()
        }

        with open(DB_FILENAME, 'w') as f:
            json.dump(master_data, f, indent=2)

    def recalculate_logbook(self):
        """Re-runs running totals from initial values."""
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
        self.save_data()
    
    def delete_flight(self, flight_id: str):
        self.flights = [f for f in self.flights if f.id != flight_id]
        self.save_data()

    def add_vor_check(self, check: VorCheck):
        self.vor_checks.append(check)
        self.save_data()

    def get_last_vor_check_date(self) -> Optional[str]:
        if not self.vor_checks:
            return None
        sorted_checks = sorted(self.vor_checks, key=lambda x: x.date, reverse=True)
        return sorted_checks[0].date

    @staticmethod
    def calculate_fms_total(start_t: time, end_t: time) -> str:
        # Simple crossing midnight logic not implemented for time objects in this snippet
        # assuming same day for simplicity, or simple subtraction
        d1 = timedelta(hours=start_t.hour, minutes=start_t.minute)
        d2 = timedelta(hours=end_t.hour, minutes=end_t.minute)
        
        if d2 < d1:
            d2 += timedelta(days=1)
            
        seconds = (d2 - d1).total_seconds()
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours:02}:{minutes:02}"

    @staticmethod
    def days_between(d1_str: str, d2_str: str) -> int:
        try:
            date1 = datetime.strptime(d1_str, "%Y-%m-%d")
            date2 = datetime.strptime(d2_str, "%Y-%m-%d")
            return abs((date1 - date2).days)
        except:
            return 0

# ==========================================
# UI RENDERING FUNCTIONS
# ==========================================

def render_sidebar(manager: FlightLogManager):
    st.sidebar.title("Polymer Aviation")
    st.sidebar.caption("Digital Flight Logbook")
    
    option = st.sidebar.radio(
        "Navigation", 
        ["Log New Flight", "Record VOR Check", "View Logbook", "Daily Summary", "Settings", "AI Assistant"],
        index=0
    )
    
    st.sidebar.divider()
    
    # Download JSON
    if os.path.exists(DB_FILENAME):
        with open(DB_FILENAME, "rb") as f:
            st.sidebar.download_button(
                "Download Master File",
                f,
                file_name="polymer_aviation_log.json",
                mime="application/json"
            )
    return option

def render_dashboard_widgets(manager: FlightLogManager):
    # Calculate current totals (Initial + All Flights)
    if manager.flights:
        last = manager.flights[-1].cumulative
        curr_hobbs = last.hobbs
        curr_lnd = last.landings
        curr_lc = last.left_engine_cycles
        curr_rc = last.right_engine_cycles
    else:
        init = manager.initial_totals
        curr_hobbs = init.hobbs
        curr_lnd = init.landings
        curr_lc = init.left_engine_cycles
        curr_rc = init.right_engine_cycles

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Hobbs", f"{curr_hobbs:.1f}", "hrs")
    c2.metric("Total Landings", curr_lnd)
    c3.metric("Left Eng. Cycles", curr_lc)
    c4.metric("Right Eng. Cycles", curr_rc)
    st.divider()

def render_log_flight_form(manager: FlightLogManager):
    st.header("✈️ Log New Flight Segment")
    
    render_dashboard_widgets(manager)

    with st.form("flight_input_form"):
        # Row 1
        c1, c2, c3, c4 = st.columns(4)
        with c1: date_input = st.date_input("Date")
        with c2: aircraft = st.selectbox("Aircraft", AIRCRAFT_OPTIONS)
        with c3: operated_by = st.text_input("Operated By", "Polymer Aviation")
        with c4: pax = st.number_input("PAX", min_value=0, step=1)
        
        # Row 2
        st.subheader("Route & Crew")
        r1, r2, r3, r4, r5 = st.columns(5)
        with r1: dep = st.text_input("Dep (ICAO)").upper()
        with r2: arr = st.text_input("Arr (ICAO)").upper()
        with r3: div = st.text_input("Alt (ICAO)").upper()
        with r4: pic = st.text_input("PIC Initials").upper()
        with r5: sic = st.text_input("SIC Initials").upper()
        
        # Row 3
        st.subheader("Times & Fuel")
        t1, t2, t3, t4, t5 = st.columns(5)
        with t1: fms_out = st.time_input("FMS Takeoff", time(12,0))
        with t2: fms_in = st.time_input("FMS Landing", time(13,0))
        with t3: hobbs_out = st.number_input("Hobbs Out", format="%.1f")
        with t4: hobbs_in = st.number_input("Hobbs In", format="%.1f")
        with t5: 
            # Datalist simulation
            fuel_opts = [x*100 for x in range(4, 41)] # 400 to 4000
            fuel = st.selectbox("Fuel Consumed (lbs)", fuel_opts, index=4) # Default 800

        # Row 4
        st.subheader("Stats")
        s1, s2, s3, s4 = st.columns(4)
        with s1: landings = st.number_input("Landings", min_value=1, value=1)
        with s2: approach = st.selectbox("Approach", APPROACH_TYPES)
        with s3: l_cyc = st.number_input("Left Eng Cycles (+)", min_value=0, value=1)
        with s4: r_cyc = st.number_input("Right Eng Cycles (+)", min_value=0, value=1)
        
        submit = st.form_submit_button("Save Flight Log", type="primary")

        if submit:
            fms_total = manager.calculate_fms_total(fms_out, fms_in)
            new_id = str(uuid.uuid4())[:8]
            
            flight = FlightSegment(
                id=new_id,
                date=date_input.strftime("%Y-%m-%d"),
                aircraft_id=aircraft,
                operated_by=operated_by,
                dep_id=dep, arr_id=arr, diversion_id=div, pax=pax,
                fms_takeoff=fms_out.strftime("%H:%M"),
                fms_landing=fms_in.strftime("%H:%M"),
                fms_total=fms_total,
                hobbs_out=hobbs_out, hobbs_in=hobbs_in,
                pic_initials=pic, sic_initials=sic,
                landings=landings, approach_type=approach,
                fuel_consumed=fuel,
                left_engine_cycles=l_cyc, right_engine_cycles=r_cyc
            )
            
            manager.add_flight(flight)
            st.success("Flight recorded successfully!")
            st.rerun()

def render_vor_form(manager: FlightLogManager):
    st.header("📡 Record Dual VOR Check")
    
    # 30-Day Check Logic
    last_date = manager.get_last_vor_check_date()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    if last_date:
        days = manager.days_between(today_str, last_date)
        if days > 30:
            st.warning(f"⚠️ **Check Expired!** Last check was {last_date} ({days} days ago).")
        else:
            st.success(f"✅ **Current.** Last check was {last_date} ({days} days ago).")
    else:
        st.info("No previous VOR checks found.")
        
    with st.form("vor_input_form"):
        date_val = st.date_input("Check Date")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### VOR #1")
            v1n = st.text_input("VOR 1 ID").upper()
            v1f = st.text_input("VOR 1 Freq")
            v1e = st.number_input("VOR 1 Error", min_value=-10, max_value=10, value=0)
        with c2:
            st.markdown("### VOR #2")
            v2n = st.text_input("VOR 2 ID").upper()
            v2f = st.text_input("VOR 2 Freq")
            v2e = st.number_input("VOR 2 Error", min_value=-10, max_value=10, value=0)
            
        st.divider()
        sig = st.selectbox("Signature", SIGNATURE_OPTIONS)
        
        submitted = st.form_submit_button("Record VOR Check")
        if submitted:
            chk = VorCheck(
                id=str(uuid.uuid4())[:8],
                flight_id="",
                date=date_val.strftime("%Y-%m-%d"),
                signature=sig,
                vor1_name=v1n, vor1_freq=v1f, vor1_error=v1e,
                vor2_name=v2n, vor2_freq=v2f, vor2_error=v2e
            )
            manager.add_vor_check(chk)
            st.success("VOR Check Recorded.")
            st.rerun()

    # VOR History Table
    st.subheader("History")
    if manager.vor_checks:
        checks_data = []
        for c in sorted(manager.vor_checks, key=lambda x:x.date, reverse=True):
            checks_data.append({
                "Date": c.date,
                "VOR 1": f"{c.vor1_name} ({c.vor1_freq})",
                "Err 1": c.vor1_error,
                "VOR 2": f"{c.vor2_name} ({c.vor2_freq})",
                "Err 2": c.vor2_error,
                "Signature": c.signature
            })
        st.dataframe(pd.DataFrame(checks_data), use_container_width=True, hide_index=True)

def render_logbook_view(manager: FlightLogManager):
    st.header("📖 Master Flight Log")
    
    if not manager.flights:
        st.info("No flights logged yet.")
        return

    # Create DataFrame for display
    # Showing newest first
    data = []
    for f in reversed(manager.flights):
        data.append({
            "Date": f.date,
            "Aircraft": f.aircraft_id,
            "Route": f"{f.dep_id} > {f.arr_id}",
            "Seg Hobbs": f"{f.segment_duration:.1f}",
            "Tot Hobbs": f"{f.cumulative.hobbs:.1f}",
            "Seg Lndg": f.landings,
            "Tot Lndg": f.cumulative.landings,
            "Fuel": f.fuel_consumed,
            "L/R Cyc": f"{f.left_engine_cycles}/{f.right_engine_cycles}",
            "Tot Cyc": f"{f.cumulative.left_engine_cycles}/{f.cumulative.right_engine_cycles}",
            "ID": f.id # Hidden column reference
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True)
    
    # Simple Delete Functionality
    with st.expander("Manage Entries"):
        del_id = st.selectbox("Select Flight to Delete (by Date/Route)", 
                              options=[f.id for f in reversed(manager.flights)],
                              format_func=lambda x: next((f"{f.date}: {f.dep_id}>{f.arr_id}" for f in manager.flights if f.id == x), x))
        if st.button("Delete Selected Flight"):
            manager.delete_flight(del_id)
            st.success("Deleted.")
            st.rerun()

def render_daily_summary(manager: FlightLogManager):
    st.header("📅 Daily Summary")
    
    date_sel = st.date_input("Select Date")
    date_str = date_sel.strftime("%Y-%m-%d")
    
    # Filter
    day_flights = [f for f in manager.flights if f.date == date_str]
    
    if not day_flights:
        st.info(f"No flights recorded for {date_str}.")
        return
    
    # Calculate Daily Totals
    t_hobbs = sum(f.segment_duration for f in day_flights)
    t_land = sum(f.landings for f in day_flights)
    t_fuel = sum(f.fuel_consumed for f in day_flights)
    t_lc = sum(f.left_engine_cycles for f in day_flights)
    t_rc = sum(f.right_engine_cycles for f in day_flights)
    ac = list(set(f.aircraft_id for f in day_flights))
    
    # Print-friendly layout
    st.markdown("---")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"## Polymer Aviation Daily Log")
        st.markdown(f"**Aircraft Flown:** {', '.join(ac)}")
    with c2:
        st.markdown(f"**Date:** {date_str}")
    
    st.markdown("### Totals")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flight Time", f"{t_hobbs:.1f}")
    m2.metric("Landings", t_land)
    m3.metric("Fuel (lbs)", t_fuel)
    m4.metric("Cycles (L/R)", f"{t_lc} / {t_rc}")
    
    st.markdown("### Segments")
    seg_data = [{
        "Route": f"{f.dep_id} > {f.arr_id}",
        "Aircraft": f.aircraft_id,
        "Time": f.segment_duration,
        "PIC": f.pic_initials
    } for f in day_flights]
    st.table(pd.DataFrame(seg_data))

def render_settings(manager: FlightLogManager):
    st.header("⚙️ Settings: Initial Totals")
    st.info("Enter the totals transferred from your previous logbook. Cumulative totals will start from here.")
    
    with st.form("settings_form"):
        i = manager.initial_totals
        nh = st.number_input("Total Hobbs", value=i.hobbs, format="%.1f")
        nl = st.number_input("Total Landings", value=i.landings)
        nlc = st.number_input("Left Engine Cycles", value=i.left_engine_cycles)
        nrc = st.number_input("Right Engine Cycles", value=i.right_engine_cycles)
        
        if st.form_submit_button("Update Initial Totals"):
            manager.initial_totals = InitialTotals(nh, nl, nlc, nrc)
            manager.save_data()
            st.success("Totals updated.")
            st.rerun()

def render_ai_assistant(manager: FlightLogManager):
    st.header("✨ AI Assistant (Gemini)")
    
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    if not api_key:
        st.warning("Please enter an API Key to use the assistant.")
        return

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    query = st.text_input("Ask about your logbook (e.g., 'How many ILS approaches last month?')")
    
    if st.button("Ask Gemini") and query:
        with st.spinner("Analyzing logbook..."):
            try:
                # Prepare context
                context = {
                    "flights": [asdict(f) for f in manager.flights],
                    "vor_checks": [asdict(v) for v in manager.vor_checks]
                }
                
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                You are an aviation logbook assistant. Here is the flight data JSON:
                {json.dumps(context, default=str)}
                
                User Question: {query}
                
                Provide a helpful, aviation-professional answer based strictly on the data provided.
                """
                
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"AI Error: {e}")

# ==========================================
# MAIN APP EXECUTION
# ==========================================

def main():
    # Initialize Logic
    manager = FlightLogManager()
    
    # Sidebar
    choice = render_sidebar(manager)
    
    # Routing
    if choice == "Log New Flight":
        render_log_flight_form(manager)
    elif choice == "Record VOR Check":
        render_vor_form(manager)
    elif choice == "View Logbook":
        render_logbook_view(manager)
    elif choice == "Daily Summary":
        render_daily_summary(manager)
    elif choice == "Settings":
        render_settings(manager)
    elif choice == "AI Assistant":
        render_ai_assistant(manager)

if __name__ == "__main__":
    main()

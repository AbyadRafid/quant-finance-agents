from agents.market_data_agent import MarketDataAgent
from agents.signal_generation_agent import SignalGenerationAgent
from agents.portfolio_optimization_agent import PortfolioOptimizationAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.macroeconomic_agent import MacroeconomicAgent
from agents.self_evaluation_agent import SelfEvaluationAgent
from agents.coordinator_agent import CoordinatorAgent

def main():
    # Initialize all specialist agents
    agents = {
        "market": MarketDataAgent(),
        "signal": SignalGenerationAgent(),
        "portfolio": PortfolioOptimizationAgent(),
        "risk": RiskManagementAgent(),
        "macro": MacroeconomicAgent(),
        "eval": SelfEvaluationAgent()
    }

    # Create and run the coordinator agent
    coordinator = CoordinatorAgent(agents)
    coordinator.run(tickers=["AAPL", "MSFT", "TLT", "GLD"])

if __name__ == "__main__":
    main()

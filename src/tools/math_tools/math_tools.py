import math
import numexpr
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Any

import requests
import xml.etree.ElementTree as ET
import urllib.parse
from langchain.tools import BaseTool

class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = """Calculate expression using Python's numexpr library.
    Expression should be a single line mathematical expression that solves the problem.
    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    
    def _run(self, expression: str) -> str:
        
        local_dict = {"pi": math.pi, "e": math.e}
        try:  
            return str(
                numexpr.evaluate(
                    expression.strip(),
                    global_dict={},  # restrict access to globals
                    local_dict=local_dict,  # add common mathematical functions
                )
            )
        except KeyboardInterrupt:
            exit(0)
        except:
            return "Error while evaluating expression!"
            
    
    async def _arun(self, expression: str) -> str: 
        local_dict = {"pi": math.pi, "e": math.e}
        return str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )

class WolframAlpha(WolframAlphaAPIWrapper):
    """Subclass of WolframAlphaAPIWrapper with modified run method."""

    def run(self, query: str) -> str:
        """Run query through WolframAlpha and parse all results."""
        res = self.wolfram_client.query(query)

        try:
            results_pod = next(
                    pod for pod in res['pod'] if pod["@title"] in ["Result", "Results", "Exact result", "Complex roots"]
                )

            if isinstance(results_pod["subpod"], dict):
                answers = [results_pod["subpod"]["plaintext"]]
            elif isinstance(results_pod["subpod"], list):
                answers = [
                    subpod["plaintext"] for subpod in results_pod["subpod"] if subpod["plaintext"]
                ]
            else:
                raise TypeError("Not supported type")

        except (StopIteration, KeyError):
            return "Wolfram Alpha wasn't able to answer it"

        if not answers:
            return "No good Wolfram Alpha Result was found"
        else:
            # Join answers into a single string
            answers_str = "\n".join(answers)
            return answers_str

    async def arun(self, query: str) -> str:
        """Run query through WolframAlpha and parse all results."""
        res = await self.wolfram_client.aquery(query)
        print(f"res: {res}")  # Debug output

        try:
            results_pod = next(
                    pod for pod in res['pod'] if pod["@title"] in ["Result", "Results", "Exact result", "Complex roots"]
                )

            if isinstance(results_pod["subpod"], dict):
                answers = [results_pod["subpod"]["plaintext"]]
            elif isinstance(results_pod["subpod"], list):
                answers = [
                    subpod["plaintext"] for subpod in results_pod["subpod"] if subpod["plaintext"]
                ]
            else:
                raise TypeError("Not supported type")

        except (StopIteration, KeyError):
            return "Wolfram Alpha wasn't able to answer it"

        if not answers:
            return "No good Wolfram Alpha Result was found"

        answers_str = "\n".join(answers)
        return answers_str

class WolframAlphaTool(BaseTool):
    name: str = "WolframAlpha"
    description: str = """
    WolframAlpha Tool: Solve complex mathematical equations and perform symbolic computations.  
    Input should be a mathematical expression or equation.
    Use this for algebra, calculus, and series sums.
    """
    # 기존 api_wrapper는 이제 직접 requests를 쓰므로 필요 없지만, 
    # 구조 유지를 위해 None으로 두거나 삭제해도 됩니다.
    api_wrapper: Optional[Any] = None 

    def _run(self, query: str) -> str:
        """에이전트가 호출할 때 실제로 실행되는 메인 로직입니다."""
        app_id = "UKHLTP2GKP"
        
        # 1. LaTeX 청소기 (Clean up)
        # 울프람이 싫어하는 기호들을 평문 수식으로 변환합니다.
        clean_arg = query.replace(r"\sum", "sum").replace(r"\infty", "infinity")
        clean_arg = clean_arg.replace(r"\frac", "").replace("{", "(").replace("}", ")")
        clean_arg = clean_arg.replace("$", "").replace("\\", "")
        
        # URL에 안전하게 포함되도록 인코딩 (띄어쓰기, 특수기호 처리)
        encoded_query = urllib.parse.quote(clean_arg)
        url = f"http://api.wolframalpha.com/v2/query?input={encoded_query}&appid={app_id}"
        
        print(f"--- WolframAlpha 호출 중: {clean_arg} ---")
        
        try:
            # 직접 API를 쏘기 때문에 라이브러리 버그(띄어쓰기 에러)를 100% 회피합니다.
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                return f"Error: WolframAlpha returned status {response.status_code}"
            
            root = ET.fromstring(response.content)
            
            # 2. 정답 Pod 낚아채기 (Infinite sum, Sum 등 범위 확장)
            # 울프람은 문제 유형에 따라 정답 제목을 다르게 줍니다.
            target_titles = ['Result', 'Infinite sum', 'Exact result', 'Value', 'Sum', 'Decimal approximation']
            
            for pod in root.findall('pod'):
                title = pod.get('title', '')
                if any(t in title for t in target_titles):
                    plaintext_node = pod.find('subpod/plaintext')
                    if plaintext_node is not None and plaintext_node.text:
                        result = plaintext_node.text
                        print(f"--- 찾은 결과: {result} ---")
                        return result
            
            return "No clear answer found. Try rephrasing the math expression."
            
        except Exception as e:
            return f"WolframAlpha Tool Error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """비동기 실행 시에도 위의 _run을 그대로 사용합니다."""
        return self._run(query)
    
class FinishTool_schema(BaseModel):
    thought: Optional[str] = Field(
        description="Thought of current status and next step based on previous messages."
    )
    answer: str = Field(
        description="""answer should be a value or latex format expression in short without any equations. Do not include any = in your answer. If you can convert the \\frac{{a}}{{b}} into simple values, answer with simplified format.
You must answer with simplified format like as follows:
-12
\\frac{{-6}}{{5}}
(1,2)""",
    )

class FinishTool(BaseTool):
    name: str = "finish"
    description: str = """Finish the question with short answer"""
    # args_schema: Type[BaseModel] = FinishTool_schema
    
    def _run(self, answer: str = "") -> str:
        return f"Answer: {answer}"

    async def _arun(self, answer: str = "") -> str:
        return f"Answer: {answer}"
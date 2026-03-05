import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from langchain_core.tools import BaseTool, tool
from typing import Literal, Any

def evaluate_webshop(output, label=None):
    ispass = False
    score = 0
    if "Your score (min 0.0, max 1.0): " in output:
        output = output
        if "Your score (min 0.0, max 1.0): 1.0" in output:
            ispass = True
        score = float(output.split("Your score (min 0.0, max 1.0): ")[-1].split()[0])
    return ispass, score

WEBSHOP_URL = "http://localhost:3000"

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
        url = (
            f'{WEBSHOP_URL}/{session}'
        )
    if page_type == 'search':
        url = (
            f'{WEBSHOP_URL}/search_results/{session}/'
            f'{query_string}/{page_num}'
        )
    elif page_type == 'item':
        url = (
            f'{WEBSHOP_URL}/item_page/{session}/'
            f'{asin}/{query_string}/{page_num}/{options}'
        )
    elif page_type == 'item_sub':
        url = (
            f'{WEBSHOP_URL}/item_sub_page/{session}/'
            f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
        )
    elif page_type == 'end':
        url = (
            f'{WEBSHOP_URL}/done/{session}/'
            f'{asin}/{options}'
        )
    # print(url)
    try:
        html = requests.get(url, timeout=5.0)
        html = html.text
    except:
        return "Request timeout. Please try again", {}
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(string=True)
    visible_texts = list(filter(tag_visible, texts))
    observation = ''
    option_type = ''
    options = {}
    asins = []
    cnt = 0
    prod_cnt = 0
    just_prod = 0
    for t in visible_texts:
        if t == '\n':
            continue
        if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '':
            continue
        if t.parent.name == 'button':  # button
            processed_t = f'\n[{t}] '
        elif t.parent.name == 'label':  # options
            if f"'{t}'" in url:
                processed_t = f'[[{t}]]'
            else:
                processed_t = f'[{t}]'
            options[str(t)] = option_type
        elif t.parent.get('class') == ["product-link"]:  # product asins
            processed_t = f'\n[{t}] '
            if prod_cnt >= 6:
                processed_t = ''
            else:
                asins.append(str(t))
            prod_cnt += 1
            just_prod = 0
        else:  # regular, unclickable text
            processed_t = '\n' + str(t) + ' '
            if cnt < 2 and page_type != 'init':
                processed_t = ''
            if just_prod <= 2 and prod_cnt >= 7:
                processed_t = ''
            option_type = str(t)
            cnt += 1
        just_prod += 1
        observation += processed_t
    info = {}
    if options:
        info['option_types'] = options
    if asins:
        info['asins'] = asins
    if 'Your score (min 0.0, max 1.0)' in visible_texts:
        idx = visible_texts.index('Your score (min 0.0, max 1.0)')
        info['reward'] = float(visible_texts[idx + 1])
        info['done'] = True
        observation = 'Your score (min 0.0, max 1.0): ' + \
            (visible_texts[idx + 1])
    if "[Search]" in observation:
            observation += "\nEnter your search query with search tool."
    return clean_str(observation), info


class webshopEnv:
    def __init__(self, session_id=None, webshop_url=WEBSHOP_URL):
        self.sessions = {}
        self.session_id = session_id
        self.url = webshop_url

    def step(self, session, action):
        try:
            if not session:
                session = self.session_id
            observation_ = None
            if action == 'reset':
                self.sessions[session] = {'session': session, 'page_type': 'init'}
            elif action.startswith('think['):
                observation = 'OK.'
            elif action.startswith('search['):
                assert self.sessions[session]['page_type'] == 'init'
                query = action[7:-1]
                self.sessions[session] = {'session': session, 'page_type': 'search',
                                        'query_string': query, 'page_num': 1}
            elif action.startswith('click['):
                button = action[6:-1]
                if button == 'Buy Now':
                    assert self.sessions[session]['page_type'] == 'item'
                    self.sessions[session]['page_type'] = 'end'
                    done = True
                elif button == 'Back to Search':
                    assert self.sessions[session]['page_type'] in [
                        'search', 'item_sub', 'item']
                    self.sessions[session] = {
                        'session': session, 'page_type': 'init'}
                elif button == 'Next':
                    assert self.sessions[session]['page_type'] == 'search'
                    self.sessions[session]['page_num'] += 1
                elif button == 'Prev':
                    assert self.sessions[session]['page_type'] in [
                        'search', 'item_sub', 'item']
                    if self.sessions[session]['page_type'] == 'search':
                        self.sessions[session]['page_num'] -= 1
                    elif self.sessions[session]['page_type'] == 'item_sub':
                        self.sessions[session]['page_type'] = 'item'
                    elif self.sessions[session]['page_type'] == 'item':
                        self.sessions[session]['page_type'] = 'search'
                        self.sessions[session]['options'] = {}
                elif button in ACTION_TO_TEMPLATE:
                    assert self.sessions[session]['page_type'] == 'item'
                    self.sessions[session]['page_type'] = 'item_sub'
                    self.sessions[session]['subpage'] = button
                else:
                    if self.sessions[session]['page_type'] == 'search':
                        assert button in self.sessions[session].get(
                            'asins', [])  # must be asins
                        self.sessions[session]['page_type'] = 'item'
                        self.sessions[session]['asin'] = button
                    elif self.sessions[session]['page_type'] == 'item':
                        assert 'option_types' in self.sessions[session]
                        assert button in self.sessions[session]['option_types'], (
                            button, self.sessions[session]['option_types'])  # must be options
                        option_type = self.sessions[session]['option_types'][button]
                        if not 'options' in self.sessions[session]:
                            self.sessions[session]['options'] = {}
                        self.sessions[session]['options'][option_type] = button
                        observation_ = f'You have clicked {button}.'
                    elif self.sessions[session]['page_type'] == 'item_sub':
                        assert button in ['Back to Search', 'Prev']
            else:
                assert False
            observation, info = webshop_text(**self.sessions[session])
            info['action'] = action
            info['observation'] = observation
            if observation_:
                observation = observation_

            self.sessions[session].update(info)
            return observation, self.sessions[session]
        except:
            if 'observation' in self.sessions[session]:
                return f"Error: Wrong tool use.\nCurrent Page:\n{self.sessions[session]['observation']}", {}
            else: 
                return f"Error: Wrong tool use.", {}

webshop_env = webshopEnv()

def set_webshop_url(url: str):
    webshop_env.url = url

class ResetTool(BaseTool):
    name: str  = "reset"
    description: str  = "Resets the webshop session to start a new search."
    session_id: str = ""

    def _run(self, session_id=None):
        if session_id:
            webshop_env.session_id = session_id
        else:
            session_id = self.session_id
            webshop_env.session_id = self.session_id
        observation, info = webshop_env.step(session_id, "reset")
        return observation.split("[Search]")[0].split('Instruction:')[-1]

class SearchTool(BaseTool):
    name: str  = "search"
    description: str  = '''search[query="the search query") - Use this tool to search for items in the WebShop. ]
    Input is a search query string.
    You can only use this tool when the observation explicitly shows a [search] button.'''
    session_id: str = None

    def _run(self, query: str):
        action = f"search[{query}]"
        observation, info = webshop_env.step(self.session_id, action)
        return (observation, info.copy())
    
    async def _arun(self, query: str):
        action = f"search[{query}]"
        observation, info = webshop_env.step(self.session_id, action)
        return (observation, info.copy())

class ClickTool(BaseTool):
    name: str  = "click"
    description: str  = """click[text="clickable button") - Use this tool to click a button in the WebShop. ]
    Clickable buttons are surrounded by []"""
    session_id: str = None

    def _run(self, text: str, thought: str = ""):
        text = text.strip(' []><')
        action = f"click[{text}]"
        observation, info = webshop_env.step(self.session_id, action)
        print(self.response_format)
        return (observation, info.copy())
    
    async def _arun(self, text: str, thought: str = ""):
        """Asynchronously run the search tool."""
        text = text.strip(' []><')
        action = f"click[{text}]"
        observation, info = webshop_env.step(self.session_id, action)
        return (observation, info.copy())

class FinishTool(BaseTool):
    name: str = "finish"
    description: str = """Finish the question with short answer"""
    
    def _run(self, answer: str = "") -> str:
        return f"Answer: {answer}", {}


    async def _arun(self, answer: str = "") -> str:
        """Asynchronously run the search tool."""
        return f"Answer: {answer}", {}
    
def revert_session(session):
    if "session" in session and session["session"] in webshop_env.sessions:
        webshop_env.sessions[session["session"]] = session


def end_condition(observation: str):
    if "Your score (min 0.0, max 1.0): " in observation:
        reward = observation.split("Your score (min 0.0, max 1.0): ")[1].split()[0]
        return True, "Your score (min 0.0, max 1.0): " + reward, float(reward)
    return False, "", 0.0

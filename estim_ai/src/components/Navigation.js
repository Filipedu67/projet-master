import React, { Component } from 'react';
import {NavLink} from 'react-router-dom';


class Navigation extends Component {

    handleClick=()=> {
        let root = document.documentElement;
        if(root.style.getPropertyValue('--background')==="#333"){
            root.style.setProperty('--color-title-1', "#4D4D4D");
            root.style.setProperty('--color-title-2', "#629C79");
            root.style.setProperty('--color-title-3', "#748BAC");
            root.style.setProperty('--background', "#313131");
            root.style.setProperty('---bg-nav-bar', "#E8DCD0");
            root.style.setProperty('--color-bg-content', "#FAEDE2");
            root.style.setProperty('--color-text-nav-sel', "#736638");
            root.style.setProperty('--color-text-nav', "#474747");
            root.style.setProperty('--color-theme', "#60D360");
            root.style.setProperty('--color-sun',"#F8B316");
            root.style.setProperty('--color-moon',"#333");
            root.style.setProperty('--shadow-moon',"none");
            root.style.setProperty('--margin-button',"24px");
        }
        else{
            root.style.setProperty('--color-title-1', "#4D4D4D");
            root.style.setProperty('--color-title-2', "#629C79");
            root.style.setProperty('--color-title-3', "#748BAC");
            root.style.setProperty('--background', "#333");
            root.style.setProperty('---bg-nav-bar', "#E8DCD0");
            root.style.setProperty('--color-bg-content', "#FAEDE2");
            root.style.setProperty('--color-text-nav-sel', "#736638");
            root.style.setProperty('--color-text-nav', "#474747");
            root.style.setProperty('--color-theme', "#333");
            root.style.setProperty('--color-moon',"#000");
            root.style.setProperty('--color-sun',"#333");
            root.style.setProperty('--shadow-moon',"1px 0px 7px white");
            root.style.setProperty('--margin-button',"2px");
        }
      }

    render() {
        return (
            <div className="sidebar">
            <div className="id">
                <div className="idContent">
                    <div className='estim'>ESTIM</div>
                    <div className='a'>A</div>
                    <div className='i'>I</div>
                </div>
            </div>
            <div className="navigationContent">
                <ul>
                    <li>
                        <NavLink exact to="" activeClassName="navActive">
                            <img src="./media/maison.png" className="maison" alt="maison"></img>
                            <span>Estimer un bien</span>
                        </NavLink>
                    </li>
                    <li>
                        <NavLink exact to="/Historique" activeClassName="navActive">
                            <img src="./media/corbeil.png" className="corbeil" alt="corbeil"></img>
                            <span>Historique</span>
                        </NavLink>
                    </li>
                    <li>
                        <NavLink exact to="/Information" activeClassName="navActive">
                            <img src="./media/info.png" className="info" alt="info"></img>
                            <span>Information</span>
                        </NavLink>
                    </li>
                </ul>
            </div>

            <div className="socialContent">
                <ul>
                    <li>
                        <a href="https://github.com/Filipedu67/projet-master" target="_blank" rel="nooperner noreferrer"><img src="./media/github.png" className="gitHub" alt="GitHub"></img></a>
                    </li>
                </ul>
            </div>
            <div className="theme">
            <i class="fas fa-moon" id="moon"></i>
                <button type="button" onClick={this.handleClick}><div id="switch_front"></div></button>
            <i class="fas fa-sun" id="sun"></i>
            </div>
        </div> 
        );
    }
}

export default Navigation;

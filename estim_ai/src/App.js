import React from 'react';
import {BrowserRouter,Route,Switch} from 'react-router-dom';
import Estimation from './pages/Estimation';
import NotFound from './pages/NotFound';
import Information from './pages/Information';
import Historique from './pages/Historique';


const App = () => {
  return (
    <>
      <BrowserRouter>
        <Switch> 
          <Route path="/Historique" exact component={Historique}/>
          <Route path="/Information" exact component={Information}/>
          <Route path="/" exact component={Estimation}/>
          <Route component={NotFound}/>
        </Switch>
      </BrowserRouter>
    </>
  );
};

export default App;
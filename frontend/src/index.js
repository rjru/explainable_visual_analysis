import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import registerServiceWorker from './registerServiceWorker';

// fetch('/wall-follow-with-tsne.json')
// fetch('/wall-follow_2018-12-12_12:16:26-with-tsne-with-sideview-with-ranking.json')
//fetch('/api/data')
fetch('/api/projection_resource')
    .then(response => response.json())
    .then((responseJSON => {
        console.log("Respuesta JSON: ", responseJSON);
        ReactDOM.render(<App data={responseJSON}/>, document.getElementById('root'));
        registerServiceWorker();
    }));

// ReactDOM.render(<App />, document.getElementById('root'));
// registerServiceWorker();

import React, {Component} from 'react';
import './App.css';
import {Layout, Row, Col} from 'antd';

import GlobalProjectionView from './components/GlobalProjectionView/GlobalProjectionView';

const {Content} = Layout;

class App extends Component {
  constructor(props) {
    super(props);
    // console.log("PROPS: ", props);
    this.state = {
      globalProjectionMatrix: props['data']['initialProjections']['pca'], // props es un json que ya tiene los datos procesados. 
    };
  }

  // Función para realizar la solicitud POST
  handleNodeClick = async (nodeData) => {
    try {
      const response = await fetch('/api/explanation_resource', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(nodeData),
      });
      const explain_node = await response.json();
      console.log(explain_node); // Procesa la respuesta del servidor según sea necesario
    } catch (error) {
      console.error('Error al realizar la solicitud:', error);
    }
  };


  render() {
    return (
        <Layout className="layout" style={{ // height: '100%', weight: '100%', height: '100%', weight: '100%', backgroundColor: 'rgba(0, 0, 0, 0)'
            }}>
            <Content style={{padding: '10px 10px', float: 'left'}}>
                <Row gutter={8} type="flex" justify="center">
                    <Col span={12} style={{ height: '100%' }}>
                      <p>PARTE 1</p>

                      <GlobalProjectionView
                        data={this.props.data}
                        globalProjectionMatrix={this.state.globalProjectionMatrix}
                        canvasHeight={486}
                        handleNodeClick={this.handleNodeClick}
                      />

                    </Col>
                    <Col span={12}>
                      <p>PARTE 2</p>
                    </Col>
                </Row>
                <Row gutter={8} style={{marginTop: 10}}>
                    <Col span={8}>
                      <p>PARTE 3</p>
                    </Col>
                    <Col span={16}>
                      <p>PARTE 4</p>
                    </Col>
                </Row>
            </Content>
        </Layout>
    );
  }

}

export default App

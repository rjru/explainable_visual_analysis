import React, {Component} from 'react';
import {findDOMNode} from 'react-dom';
import * as d3 from 'd3';
import d3Tip from 'd3-tip';
import numeric from 'numericjs';
// import isEqual from 'lodash/isEqual';
//import {Menu, Checkbox, Icon} from 'antd';
import {labelNormalColorMap} from '../../utils/Color';

import '../../styles/scatterplot.css';

const CANVAS_PADDING = 10;

//const ACTIVATED_POINT_OPACITY = 0.85;


export default class Scatterplot extends Component {

    constructor(props) {
        super(props);

        const {dataVectors, projectionMatrix} = this.props;
        /**
         * Initialize scales
         */
        const coordinates = projectionMatrix 

        this.state = {
            coordinates: coordinates,
        };
    }


    componentDidMount() {
        /**
         * Compute the points that should be disabled because of highlighting
         */
        const svgRoot = this.svgRoot = d3.select(findDOMNode(this)).select('svg');

        svgRoot.call(
            d3.zoom()
                .on('zoom', function () {
                    svgRoot.select('g#base-group')
                        .attr('transform', d3.zoomTransform(this));
                })
        );

        const {xScale, yScale} = this.computeLinearScales(this.state.coordinates);

        this.initializeScatterplot({
            xScale: xScale,
            yScale: yScale,
            coordinates: this.state.coordinates,
        });

        this.setState({xScale, yScale});
    }

    computeLinearScales(coordinates) {
        // const halfCanvasWidth = this.props.canvasWidth / 2,
        const halfCanvasHeight = this.props.canvasHeight / 2,
            halfCanvasWidth = this.svgRoot.node().getBoundingClientRect().width / 2;

        const coorT = numeric.transpose(coordinates);

        return {
            xScale: d3.scaleLinear()
                .domain(d3.extent(coorT[0]))
                .range([-halfCanvasWidth + CANVAS_PADDING, halfCanvasWidth - CANVAS_PADDING]),
            yScale: d3.scaleLinear()
                .domain(d3.extent(coorT[1]))
                .range([halfCanvasHeight - CANVAS_PADDING, -halfCanvasHeight + CANVAS_PADDING])
        };
    }

    initializeScatterplot({xScale, yScale, coordinates}) {
        // const halfCanvasWidth = this.props.canvasWidth / 2,
        const halfCanvasHeight = this.props.canvasHeight / 2,
            halfCanvasWidth = this.svgRoot.node().getBoundingClientRect().width / 2;

        const {label, labelNames} = this.props;

        // const coordinates = this.state.coordinates;
        const svgRoot = this.svgRoot,
            baseGroup = svgRoot.select('g#base-group');

        const tooltip = d3Tip()
            .attr('class', 'd3-tip-scatterplot')
            .html(d => `<p>Instance: ${d.index}</p><p style="margin-bottom:0">Label: <span style="color:${labelNormalColorMap[d.label]}">${labelNames[d.label]}</span></p>`)
            .direction('n')
            .offset([-3, 0]);

        const dragHandler = d3.drag()
            .on('start', function(d, i) {
                // Opcional: Define lo que sucede al empezar a arrastrar
                d3.select(this).raise().classed('active', true);
            })
            .on('drag', function(d, i) {
                // Actualiza la posición del punto basado en el arrastre
                const dx = d3.event.dx;
                const dy = d3.event.dy;
                const x = parseFloat(d3.select(this).attr("cx")) + dx;
                const y = parseFloat(d3.select(this).attr("cy")) + dy;
                
                // Actualizar la posición del punto basado en el movimiento del drag
                d3.select(this).attr("cx", x).attr("cy", y);
            })
            .on('end', function(d, i) {
                // Opcional: Define lo que sucede al terminar de arrastrar
                d3.select(this).classed('active', false);
                const x = parseFloat(d3.select(this).attr("cx")) 
                const y = parseFloat(d3.select(this).attr("cy")) 
                console.log("nuevos puntos: ", x," , ", y)
                // Aquí podrías, por ejemplo, actualizar el estado de React con la nueva posición del punto,
                // pero ten cuidado con cómo manejas el estado para evitar conflictos entre D3 y React
            });

        baseGroup.select('g#point-group').selectAll('.point').remove();
        baseGroup.call(tooltip);

        svgRoot.on('click', () => {
        }).on('dblclick.zoom', null);

        baseGroup.select('g#point-group')
            .attr('transform', 'translate(' + halfCanvasWidth + ',' + halfCanvasHeight + ')')
            .selectAll('.point')
            .data(coordinates)
            .enter()
            .append('circle')
            .attr('id', (d, i) => `point-${i}`)
            .attr('class', (d, i) => 'class-label-' + label[i])
            .classed('point', true)
            .attr('r', 3.5)
            .attr('cx', d => xScale(d[0]))
            .attr('cy', d => yScale(d[1]))
            .style('fill', (d, i) => labelNormalColorMap[label[i]])
            .on('mouseenter', (d, i, n) => {
                tooltip.show({
                    index: i,
                    label: label[i]
                }, n[i]);
            })
            .on('mouseleave', (d, i, n) => {
                tooltip.hide({
                    index: i,
                    label: label[i]
                }, n[i]);
            })
            .on('click', (d, i, nodes) => {
                // los valores de d y coordinates son los mismos. Esta proyección ha sido multiplicado el vector original por los datos proyectados. 
                console.log("d: ", d, " i: ", i); // EL VALOR D tiene los valores x, y. El indice lo tiene i
                console.log("Coordinates: ", coordinates[i]);
                // se ha quitado una parte de la propuesta original y ahora el código está más simple, los valores de d y coordinates coinciden con PROPS orignales. 
                // Estos son los datos originales proyectados. 
                console.log("PROPS ->: ", this.props['projectionMatrix'][i]);

                // Obtiene el elemento SVG del punto sobre el que se hizo click
                const clickedPoint = d3.select(nodes[i]);
                // Obtiene los valores de las coordenadas del punto
                const cx = clickedPoint.attr("cx");
                const cy = clickedPoint.attr("cy");
                console.log("cx: ", cx, ", cy: ", cy);

                console.log("cx inverted: ", xScale.invert(cx), ", cy inverted: ", yScale.invert(cy));

                // Aquí puedes construir el objeto nodeData basado en la información del nodo clickeado
                const nodeData = {
                    index: i,
                    coordinates: [xScale.invert(cx), yScale.invert(cy)],
                    // Agrega cualquier otra información relevante sobre el nodo
                };
        
                // Llama a la función handleNodeClick pasada como prop
                this.props.handleNodeClick(nodeData);

                // Muestra los valores en la consola
                // Los datos de dscale y cx,cy son los mismos y parten de los datos ya procesados y luego escalados. 
                console.log("d SCALE ->: ", xScale(d[0]), " - ", yScale(d[1]));

                d3.event.stopPropagation();
            })
            .call(dragHandler);
    }


    render() {
        return (
            <div
                // style={{background: '#fff', width: this.props.canvasWidth, height: this.props.canvasHeight}}
                style={{
                    background: '#fff',
                    border: '1px solid #e8e8e8',
                    borderRadius: '1px',
                    height: this.props.canvasHeight + 2
                }}
            >
                <svg
                    id={'scatterplot-canvas'}
                    //width={this.props.canvasWidth}
                    height={this.props.canvasHeight}
                    width="100%"
                    // height="100%"
                >
                    <g id="base-group">
                        <g id="point-group"/>
                    </g>
                </svg>
            </div>
        );
    }
}
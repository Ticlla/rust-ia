import React from 'react';
import { Composition } from 'remotion';
import { Clase01Video } from './Clase01Video';

export const Root = () => {
  return (
    <Composition
      id="Clase01"
      component={Clase01Video}
      durationInFrames={360}
      fps={30}
      width={1080}
      height={1920}
    />
  );
};

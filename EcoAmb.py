import matplotlib.pyplot as plt
import numpy as np
#import scipy.optimize as sp_opt

class Consumer:
    
    import scipy.optimize as sp_opt
    
    ###############################################
    ### Inicializa los argumentos de la clase  ###
    ###############################################
    
    def __init__(self,U, DU_1 = None, DU_2 = None, Indiff_Set = None,
                 D_Analytic = None, Cmin = 0.1, Cmax = 10, N = 50, I = 2,
                 Pmin = 0.05, Pmax = 12):
        
        self.U = U # Función de utilidad
        self.I = I # Ingresos
        self.h = 0.00001 # Aproximación de las derivadas numéricas
        
        #DU_1 = Derivada parcial del bien 1. Si no se explicita, se calcula la derivada de forma numérica
        if DU_1 is None:
            self.DU_1 = lambda c1,c2: (self.U(c1+self.h,c2) - self.U(c1-self.h,c2))/(2*self.h)
        else:
            self.DU_1 = DU_1
        
        #DU_2 = Derivada parcial del bien 2. Si no se explicita, se calcula la derivada de forma numérica
        if DU_2 is None:
            self.DU_2 = lambda c1,c2: (self.U(c1,c2+self.h) - self.U(c1,c2-self.h))/(2*self.h)
        else:
            self.DU_2 = DU_2
        
        self.D_Analytic = D_Analytic # Función de demanda
        self.Indiff_Set = Indiff_Set # Curva de indiferencia
        self.Cmin = Cmin # Cantidad de consumo mínima para hacer gráficos
        self.Cmax = Cmax # Cantidad de consumo máxima para hacer gráficos
        self.N = N # Número de puntos del gráfico
        self.Pmin = Pmin # Precio mínimo para hacer gráficos
        self.Pmax = Pmax # Precio máximo para hacer gráficos
                
            
            
    ###############################################
    ### Función que genera los conjuntos ##########
    ### de indiferencia ###########################
    ###############################################
    
    def C2_Indiff_Set(self, u, C1 = None):
        """
        Entrega consumo del bien 2 C2 tal que 
        u(C1,C2) = u, donde 
        
        u -> nivel de utilidad
        C1 -> Consumo del bien 1
            Si C1 es vacío, se interpreta que C1 es un vector de cantidades [Cmin, Cmax]
        
        """

        if C1 is None:
            C1 = np.linspace(self.Cmin,self.Cmax,self.N)


        if self.Indiff_Set is None:
            
            def c2_solve(u,c1):
                return self.sp_opt.fsolve(lambda c2: u - self.U(c1,c2),1/c1)[0]
            
            C2 = [c2_solve(u,c1) for c1 in C1]

        else:
            C2 = [self.Indiff_Set(u,c1) for c1 in C1]

        return C2
    
    
    ###############################################
    ### Función que genera la pendiente ###########
    ### de la curva de indiferencia ###############
    ###############################################
    
    def Indiff_Set_Slope(self,c1,c2):
        """
        Entrega la pendiente de la curva de nivel en el punto (C1,C2)
        c1-> Consumo del bien 1
        c2-> Consumo del bien 2
        """
        return -self.DU_1(c1,c2)/self.DU_2(c1,c2)


    ###############################################
    ### Función que calcula la demanda por###########
    ### ambos bienes ###############
    ###############################################
    def D(self, p1, p2, I):
        """
        Calcula la demanda
        p1-> precio bien 1
        p2-> precio bien 2
        I -> ingreso disponible
        
        Output es una lista con el consumo de cada bien
        D -> [x1, x2] 
        """
        
        # Punto inicial [x1, x2] para el algoritmo de optimización 
        xi = [I/np.max([p1,p2]), I/np.max([p1,p2])]
        
        # Negativo de la utilidad, ya que el algoritmo de optimización busca el mínimo
        def negU(x):
            return -self.U(x[0],x[1])
        
        # Restricción presupuestaria
        def Budget(x):
            return np.array([x[0]*p1+x[1]*p2 - I])
        
        # Si no se entrega una solución analítica se calcula numéricamente
        if self.D_Analytic is None:
            MP = self.sp_opt.minimize(negU, xi, constraints = ({'type':'eq', 'fun':Budget}))
            X = MP.x

        else:
            X = self.D_Analytic(p1,p2,I)
        
        # Se verifica que la solución sea positiva
        # En caso de que una solución sea negativa, se imputa cero y se consume todo el ingreso en el otro bien
        
        V = (np.array(X) >=  [0,0])
        v = (np.sum(V) == 2)

        return X * (v) + (1-v) * (V * [I/p1,I/p2])

    
    
    ###############################################
    ### Función que grafica la demanda ###########
    ### del bien 1###############
    ###############################################
    def PlotDemanda(self,p2,I):
        """
        Se grafica la demanda por el bien 1 en el plano
        (p1,Q)
        
        p2-> precio del bien 2
        I -> Ingreso disponible
        """
        
        # Precisión gráfico
        N = 50 
        
        # Grilla precios
        P1 = np.linspace(self.Pmin,self.Pmax,N)

        p2_b = p2
        I_b = I
        
        # Se calcula la demanda usando la función definida anteriormente
        X1 = [self.D(p1,p2_b,I_b)[0] for p1 in P1]
        
        # Gráfico
        
        fig, axes = plt.subplots(figsize=(5.5,5.5))
        axes.plot(X1,P1)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        textstr = '\n'.join((
            r'$p_2=%.2f$' % (p2, ),
            r'$I = %.2f$' % (I,)))
        axes.text(0.05, 0.95, textstr, transform=axes.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.xlabel(r'Bien 1')
        plt.ylabel(r'Precio Bien 1')
        plt.xlim(0,self.Cmax)
        plt.ylim(0,self.Pmax)
        plt.title(r'Demanda por el bien 1')
        plt.rc('font', size=15)
        
    ###############################################
    ### Función que mapa de ###########
    ### curvas de indiferencias ###############
    ###############################################
    def Indiff_Map_Plot(self,u):
        """
        u -> lista o vector con niveles de utilidad a graficar
        """
        
        C1 = np.linspace(self.Cmin,self.Cmax,self.N)
        
        fig, axes = plt.subplots(figsize = (5.5,5.5))
        for ui in u:
            axes.plot(C1, self.C2_Indiff_Set(ui, C1), label = r'U = %.2f'% (ui,))
        plt.xlabel(r'Bien 1')
        plt.ylabel(r'Bien 2')
        plt.xlim(0,self.Cmax)
        plt.ylim(0,self.Cmax)
        plt.legend( loc='upper right', borderaxespad=0.5)
        plt.title(r'Mapa de curvas de Indiferencia')
        plt.rc('font', size=15)


    ###############################################
    ### Función que grafica la recta tangente ###########
    ### a la curva de nivel u en c1 ###############
    ###############################################     
    def TMGS(self,c1,u):
        """
        c1-> Consumo del bien 1
        u -> Nivel de utilidad
        """
        #C1.Indiff_Map_Plot(np.array([u]))
        c2 = self.C2_Indiff_Set(u,[c1])[0]
        m  = self.Indiff_Set_Slope(c1,c2)
        def line(x):
            return m*x +(c2-m*c1) 
        xrange = np.linspace(c1-2, c1+2, 10)
        # Rango en el que se graficará la recta tangente
        # La recta tangente se gráfica en un radio 2 c/r a x1
        plt.plot(xrange, line(xrange), 'C1--')
        plt.scatter(c1,c2)
        

    ###############################################
    ### Función que grafica el consumo,  ###########
    ### la curva de nivel y la recta presupuestaria ###
    ############################################### 
    
    def No_Opt_consumo_Plot(self,c1,c2,p1,p2,I):
        """
        c1 -> Consumo del bien 1
        c2 -> Consumo del bien 2
        p1 -> Precio del bien 1
        p2 -> Precio del bien 2
        I  -> Ingreso disponible
        """
        
        # Restricción presupuestaria
        def recta_presupuestaria(c,p1,p2,I):
            return I/p2 -c*(p1/p2)
        
        # Nivel de utilidad de consumir (c1,c2)
        u = self.U(c1,c2)
        
        # Grilla consumo del bien 1
        C1 = np.linspace(self.Cmin,self.Cmax,self.N)
        
        # Puntos restricción presupuestaria
        C2_P = recta_presupuestaria(C1,p1,p2,I)
        
        # Puntos curva de indiferencia
        C2_I = self.C2_Indiff_Set(u, C1)
        
        
        # Gráfico
        fig, axes = plt.subplots(figsize = (7,7))
        
        axes.plot(C1,C2_P)
        axes.fill_between(C1,C2_P, color='C0', alpha=0.2)
        axes.plot(C1,C2_I) 
        axes.scatter(c1,c2, color = 'red')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join((
            r'$p1/p2=%.2f$' % (p1/p2, ),
            r'$(c_1,c_2) = (%.2f,%.2f)$' % (c1,c2,),
            r'$u(c_1,c_2)=%.2f$' % (u, )))
        axes.text(0.05, 0.95, textstr, transform=axes.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        axes.plot(np.linspace(c1,c1,self.N),np.linspace(0,c2,self.N), 'C1--')
        axes.plot(np.linspace(0,c1,self.N),np.linspace(c2,c2,self.N), 'C1--')
        plt.ylim(self.Cmin,self.Cmax)
        plt.xlabel(r'Bien 1')
        plt.ylabel(r'Bien 2')
        plt.title('Consumo')
        
    ###############################################
    ### Función que grafica el nivel óptimo de consumo,  
    ### la curva de nivel y la recta presupuestaria ###
    ############################################### 
    def Opt_consumo_Plot(self,p1,p2,I):
        """
        p1 -> Precio del bien 1
        p2 -> Precio del bien 2
        I  -> Ingreso disponible
        """
        def recta_presupuestaria(c,p1,p2,I):
            return I/p2 -c*(p1/p2)
        
        c1 = self.D(p1,p2,I)[0]
        c2 = self.D(p1,p2,I)[1]
        
        self.No_Opt_consumo_Plot(c1,c2,p1,p2,I)
        

class Market:
    
    import scipy.optimize as sp_opt
    import scipy.integrate as integrate
    
    def __init__(self, Pd, Ps, td = 0, ts = 0, Pr = np.inf, Pr_Max_Min = 1,
                 Pmin = 0, Pmax = 2, Qmin = 0, Qmax = 2,
                 Extm_P = lambda Q: 0, Extm_C = lambda Q: 0):
        self.Pd = Pd # Función inversa de demanda
        self.Ps = Ps # Función inversa de oferta
        self.td = td # Impuesto (subsidio) a la demanda
        self.ts = ts # Impuesto (subsidio) a la oferta
        self.Pr = Pr # Precio regulado
        self.Pr_Max_Min = Pr_Max_Min # Si el precio regulado es máximo o mínimo
        self.Pmax = Pmax # Precio máximo en el gráfico
        self.Pmin = Pmin # Precio mínimo en el gráfico
        self.Qmax = Qmax # Cantidad máximo en el gráfico
        self.Qmin = Qmin # Cantidad mínima en el gráfico
        self.N = 100 # Número de puntos por gráfico
        self.Extm_P = Extm_P # Externalidad marginal de la producción
        self.Extm_C = Extm_C # Externalidad marginal del consumo
        
    def Equilibrio(self):
        """
        Encuentra el equilibrio en este mercado, tomando en cuenta los 
        eventuales impuestos, subsidios y regulaciones de precio.
        """
        # Equilibrio de mercado ignorando precios mínimos o máximos #
        Qm = self.sp_opt.fsolve(lambda Q: (self.Pd(Q) - self.td)  -
                                (self.Ps(Q) + self.ts),0.1)[0]
        Pm = self.Pd(Qm)
        
        # Se verifica si el precio de mercado supera o no el precio máximo #
        if (self.Pr_Max_Min == 1 or self.Pr_Max_Min == 'Max'):
            P_ = np.min([self.Pr,Pm])
            
        # Se verifica si el precio de mercado es menor al precio mínimo #
        elif (self.Pr_Max_Min == 0 or self.Pr_Max_Min == 'Min'):
            P_ = np.max([self.Pr,Pm])
            
        # Si ambos precios no difieren se toma el precio de mercado #
        if np.abs((P_ - Pm))/Pm < 0.0001:
            return [Qm,P_]
        
        # Si difieren se toma el menor #
        else:
            Q_d = self.sp_opt.fsolve(lambda Q:(self.Pd(Q) - self.td) - P_,0.1)
            Q_s = self.sp_opt.fsolve(lambda Q:(self.Ps(Q) + self.ts) - P_,0.1)
            Q_ = np.min([Q_d,Q_s])
            return [Q_,P_]
    

    
    def CMGS(self,Q):
        """
        Función que encuentra el costo marginal social. En ausencia 
        de externalidades, este costo es igual al costo marginal
        privado.
        CMGS = CMGP + EXTM
        """
        return self.Ps(Q) + self.Extm_P(Q)
    
    def BMGS(self,Q):
        """Función que encuentra el beneficio marginal social. En ausencia 
        de externalidades, este beneficio es igual a la disposición a pagar.
        BMGS = D + EXTM"""
        return self.Pd(Q) + self.Extm_C(Q)
    
    def Plot_Equilibrio(self, mostrar_excedente = False,
                       mostrar_EC = False, # Mostrar el excedente del consumidor
                       mostrar_EP = False, # Mostrar el excedente del productor
                       mostrar_Gasto = False, # Mostrar gasto/ingreso del gobierno 
                       mostrar_DWL = False, # Mostrar DWL
                       mostrar_CMS = False,
                       mostrar_BMS = False,
                       titulo = r'Equilibrio de mercado'): 
        """
        Grafica el equilibrio de mercado
        """
        
        #P = np.linspace(self.Pmin,self.Pmax,self.N) # Precios a graficar
        Q = np.linspace(self.Qmin,self.Qmax,self.N) # Cantidades a graficar
        
        plt.plot(Q, self.Pd(Q), color = 'blue') # Grafico demanda
        plt.plot(Q, self.Ps(Q), color = 'red')  # Grafico oferta
        
        Q_, P_ = self.Equilibrio() # Se computa el equilibrio
        
        locate_dot(Q_, P_) # Grafica el punto de equilibrio
        
        if mostrar_excedente == True:
            
            mostrar_EC = True
            mostrar_EP = True
            mostrar_Gasto = True
            mostrar_DWL = True
            
            
            
        if self.ts != 0:
            plt.plot(Q, self.Ps(Q) + self.ts, color = 'darkred')
            
            locate_dot(Q_,P_ - self.ts)
            
            
        if self.td != 0:
            plt.plot(Q, self.Pd(Q) - self.td, color = 'darkblue')
            
            locate_dot(Q_,P_ - self.td)
            
            
        Q_social = self.sp_opt.fsolve(lambda Q: (self.BMGS(Q) )  -
                            (self.CMGS(Q)),0.1)[0]
        P_social = self.CMGS(Q_social)
            
        if self.Extm_P(Q_) != 0:
            
        
            plt.plot(Q, self.CMGS(Q), color = 'darkgreen')
            locate_dot(Q_social,P_social)
            
        if self.Extm_C(Q_) != 0:
            plt.plot(Q, self.BMGS(Q), color = 'darkgreen')
            locate_dot(Q_social,P_social)
            
        if mostrar_EC == True:
            Qh = np.linspace(0,Q_,int(self.N/2))

            plt.fill_between(Qh,self.Pd(Qh), np.linspace(P_,P_,int(self.N/2)),
                                    color = 'blue', alpha = 0.3)               
                
        if mostrar_EP == True:
            Qh = np.linspace(0,Q_,int(self.N/2))
                
            plt.fill_between(Qh, np.linspace(P_ -self.ts,P_-self.ts,int(self.N/2)), self.Ps(Qh),
                                 color = 'red', alpha = 0.3)
                
        if mostrar_Gasto == True:
            Qh = np.linspace(0,Q_,int(self.N/2))
                
            plt.fill_between(Qh, np.linspace(P_ -self.ts,P_-self.ts,int(self.N/2)), 
                                np.linspace(P_,P_,int(self.N/2)),
                                color = 'green', alpha = 0.3)
        if mostrar_CMS == True:
            Qh = np.linspace(0,Q_, int(self.N/2))
            
            plt.fill_between(Qh, self.CMGS(Qh),
                                  color = 'green', alpha =0.3)
        if mostrar_BMS == True:
            Qh = np.linspace(0,Q_, int(self.N/2))
            
            plt.fill_between(Qh, self.BMGS(Qh),
                                  color = 'green', alpha =0.3)
        
        if mostrar_DWL == True: 
            
            
            
            """Q_aux = self.sp_opt.fsolve(lambda Q: (self.Pd(Q))  -
                                (self.Ps(Q)),0.1)[0]"""
            plt.scatter(Q_social,self.CMGS(Q_social))
            if Q_social < Q_:
                Qi = np.linspace(Q_social,Q_,int(self.N/2))
                plt.fill_between(Qi,  self.CMGS(Qi),self.BMGS(Qi),
                                    color = 'gray', alpha = 0.3)
            else:
                Qi = np.linspace(Q_,Q_social,int(self.N/2))
                plt.fill_between(Qi, self.Pd(Qi), self.Ps(Qi),
                                      color = 'gray', alpha = 0.3)
                    
        if self.Pr < np.inf:
            Q_auxd = self.sp_opt.fsolve(lambda Q: (self.Pd(Q))  -
                                    self.Pr,0.1)[0]
            plt.scatter(Q_auxd,self.Pr, color = 'red')
            
            plt.plot(np.linspace(0,Q_auxd,int(self.N/2)),
                          np.linspace(self.Pr,self.Pr,int(self.N/2)),
                          linestyle = '-', color = 'black')
            
            plt.plot(np.linspace(Q_auxd,Q_auxd,int(self.N/2)),
                          np.linspace(0,self.Pr,int(self.N/2)),
                          linestyle = '-.', color = 'green')
            
            
            
            Q_auxs = self.sp_opt.fsolve(lambda Q: (self.Ps(Q))  -
                                    self.Pr,0.1)[0]
            plt.scatter(Q_auxs,self.Pr, color = 'red')
            
            plt.plot(np.linspace(0,Q_auxs,int(self.N/2)),
              np.linspace(self.Pr,self.Pr,int(self.N/2)),
              linestyle = '-', color = 'black')
            plt.plot(np.linspace(Q_auxs,Q_auxs,int(self.N/2)),
                          np.linspace(0,self.Pr,int(self.N/2)),
                          linestyle = '-.', color = 'green')
            
            Qm = np.min([Q_auxd,Q_auxs])
            if mostrar_EC == True:
                Qh = np.linspace(0,Qm,int(self.N/2))

                plt.fill_between(Qh,self.Pd(Qh), np.linspace(self.Pr,self.Pr,int(self.N/2)),
                                     color = 'blue', alpha = 0.3)
            if mostrar_EP == True:
                Qh = np.linspace(0,Qm,int(self.N/2))

                plt.fill_between(Qh,np.linspace(self.Pr,self.Pr,int(self.N/2)),
                                      self.Ps(Qh), 
                                     color = 'red', alpha = 0.3)
            if mostrar_DWL == True:
                Qh = np.linspace(Qm,Q_,int(self.N/2))
                plt.fill_between(Qh,self.Pd(Qh),self.Ps(Qh), color = 'gray', alpha = 0.3)
                
        
        plt.ylim(0, self.Pmax)
        plt.xlim(0, self.Qmax)
        plt.xlabel(r'Cantidad')
        plt.ylabel(r'Precio')
        plt.title(titulo)
        
    def Tax_Payment(self):
        """
        Encuentra el eventual pago de impuesto o los eventuales ingresos por subsidios
        """
        return [self.Equilibrio()[0]*self.td,
                self.Equilibrio()[0]*self.ts]
    
    def CET(self, Q_ = None, P_ = None):
        if Q_ == None or P_ == None:
            Q_social = self.sp_opt.fsolve(lambda Q: (self.BMGS(Q) )  -
                            (self.CMGS(Q)),0.1)[0]
        
        
        return self.integrate.quad(self.Extm_P, 0, Q_social)[0]
        
    def BET(self, Q_ = None, P_ = None):
        if Q_ == None or P_ == None:
            Q_social = self.sp_opt.fsolve(lambda Q: (self.BMGS(Q) )  -
                            (self.CMGS(Q)),0.1)[0]
        return self.integrate.quad(self.Extm_C, 0, Q_social)[0]

    
    def Ex_C(self, Q_ = None, P_ = None):
        
        if Q_ == None or P_ == None:
            Q_, P_ = self.Equilibrio()
        
        return self.integrate.quad(self.Pd, 0, Q_)[0] - Q_*P_ - self.Tax_Payment()[0]

    def Ex_P(self, Q_ = None, P_ = None):
        
        if Q_ == None or P_ == None:
            Q_, P_ = self.Equilibrio()
            
        return   Q_*P_ - self.integrate.quad(self.Ps, 0, Q_)[0] - self.Tax_Payment()[1]
    

        
        
class Producer:
    
    import scipy.optimize as sp_opt
    import scipy.integrate as integrate
    
    def __init__(self,CT, CMG = None, O_Analytic = None,
                 t = 0.0,
                 l = 0.0,
                 T = 0.0,
                 Qmin = 0.1, Qmax = 10, N = 50,
                 Pmin = 0.05, Pmax = 12):
        
        self.CT = CT
        self.h = 0.000000001
        if CMG is None:
            self.CMG = lambda Q: (self.CT(Q + self.h) -self.CT(Q -self.h))/(2*self.h)
        else:
            self.CMG = CMG
        self.O_Analytic = O_Analytic
        self.Qmin = Qmin
        self.Qmax = Qmax
        self.N = N
        self.Pmin = Pmin
        self.Pmax = Pmax
        self.t = t
        self.l = l
        self.T = T
        self.Pi = lambda P,Q: (P -self.t)*Q - self.CT(Q) - self.T
        self.CME = lambda Q: self.CT(Q)/Q
    
    def Qmin_O(self):
        Q_ =  self.sp_opt.fsolve(lambda Q: (self.CMG(Q)*(1+self.l) +self.t) - ((self.CT(Q) +self.t*Q + self.CMG(Q)*Q*self.l +self.T )/Q  ), 0.01)[0]
        P_ = self.CMG(Q_)
                                 
        return [Q_,P_]
    
    def Plot_Oferta(self):
        
        Q_ = self.Qmin_O()[0]
        #P_ = self.Qmin_O()[1]
        
        Q  = np.linspace(Q_,self.Qmax,self.N)
        Ps = (self.CMG(Q) + self.t)/(1-self.l)
        plt.plot(Q,Ps)
        Qn = np.linspace(0,Q_, int(self.N/2))
        Psn= (self.CMG(Qn) +self.t)/(1 -self.l)
        plt.plot(Qn,Psn, linestyle = 'dotted')
        plt.xlim(self.Qmin,self.Qmax)


class FPP:
    
    import scipy.optimize as sp_opt
    
    def __init__(self, u1 = None, u2 = None,
                 Xmax = 10, N = 30):
        
        if u1 is None:
            self.u1 = lambda x: x**0.5
        else:
            self.u1 = u1
        if u2 is None:
            self.u2 = lambda x: x**0.5 
        else:
            self.u2 = u2
        
        self.Xmax = Xmax
        self.N = N
    
    def Plot_FPP(self, mostrar_factible = False,x = None):
        
        X1 = np.linspace(0.1,self.Xmax,self.N)
        U1 = self.u1(X1)
        

        X2 = self.Xmax- X1
        U2 = self.u2(X2)   
        
        plt.plot(U1,U2)
        plt.fill_between(U1,U2, alpha = 0.5)
        u_max = np.max([self.u1(self.Xmax),self.u2(self.Xmax)])
        if x is not None:
            locate_dot(x[0],x[1])
        plt.xlim(0,u_max)
        plt.ylim(0,u_max)
        plt.xlabel(r'Utilidad individuo 1')
        plt.ylabel(r'Utilidad individuo 2')
        plt.title(r'Frontera de posibilidades de utilidad')


def locate_dot(Q_, P_, show_line = True,
               color1 = 'green',
               annotate = None,
               color2 = None):
    """
    Resalta el punto (Q_,P_) en un gráfico
    """
    N = 10
    plt.scatter(Q_,P_, color = color1)
    if show_line == True:
        plt.plot(np.linspace(0,Q_,int(N/2)),
                        np.linspace(P_,P_,int(N/2)),
                     linestyle = '-.', color = color1)
        
        plt.plot(np.linspace(Q_,Q_,int(N/2)),
                    np.linspace(0,P_,int(N/2)),
                     linestyle = '-.', color = color1)
    
    if color2 is None:
        color2 = color1
        
    if annotate is not None:
        plt.annotate(annotate, (Q_,P_),bbox={'facecolor': color2, 'alpha': 0.3, 'pad': 10},fontsize=15)
        
def plot_slope(x0,y0,m, l = 0.3, N = 25):
    c = y0 - x0*m
    def line(x):
        return x*m + c
    
    Xh = np.linspace(x0*(1-l),x0*(1+l),N) 
    Yh = line(Xh)
    plt.scatter(x0,y0, c='red')
    plt.plot(Xh,Yh, linestyle = '-.')